/*
 * Case 04 — Phase 5: Official DiskANN3 Search CLI Adapter
 *
 * Integrates Microsoft DiskANN3 disk index search engine. Loads persisted
 * index artifacts, executes approximate nearest-neighbor search, and extracts
 * candidate local IDs and exact squared-L2 distances with optional global ID mapping.
 */

use std::fs::File;
use std::io::{BufReader, BufWriter, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use diskann_disk::{
    data_model::{AdHoc, CachingStrategy},
    search::{
        provider::{
            disk_provider::DiskIndexSearcher,
            disk_vertex_provider_factory::DiskVertexProviderFactory,
        },
        search_mode::SearchMode,
    },
    storage::disk_index_reader::DiskIndexReader,
};
use diskann_providers::storage::{
    get_compressed_pq_file, get_disk_index_file, get_pq_pivot_file, FileStorageProvider,
};
use diskann_vector::distance::Metric;

#[derive(Parser, Debug)]
#[command(name = "diskann_query", about = "Query official DiskANN3 disk-backed index")]
struct Args {
    /// Path prefix of the index (without _disk.index extension)
    #[arg(long)]
    index_prefix: PathBuf,

    /// Path to queries binary file (.bin in DiskANN format)
    #[arg(long)]
    queries: Option<PathBuf>,

    /// Raw JSON query array of floats (e.g. "[0.1, 0.2, ...]")
    #[arg(long)]
    single_query_json: Option<String>,

    /// Number of nearest neighbors to retrieve (K)
    #[arg(short = 'k', long, default_value_t = 10)]
    top_k: usize,

    /// Search list size (L)
    #[arg(short = 'l', long, default_value_t = 100)]
    search_l: u32,

    /// Beam width for parallel IO traversal
    #[arg(long, default_value_t = 4)]
    beam_width: usize,

    /// Number of search threads
    #[arg(long, default_value_t = 4)]
    num_threads: usize,

    /// Number of graph nodes to cache in RAM
    #[arg(long, default_value_t = 0)]
    cache_nodes: usize,

    /// Optional path to id_mapping.json
    #[arg(long)]
    id_map: Option<PathBuf>,

    /// Path to save output JSON
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Serialize, Deserialize, Debug)]
struct IdMappingFile {
    cold_vector_count: usize,
    first_cold_id: usize,
    last_cold_id: usize,
    local_to_global: Vec<usize>,
}

#[derive(Serialize, Deserialize, Debug)]
struct QueryResultItem {
    query_idx: usize,
    local_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    global_ids: Option<Vec<usize>>,
    distances: Vec<f32>,
    duration_micros: u64,
}

#[derive(Serialize, Deserialize, Debug)]
struct OutputReport {
    index_prefix: String,
    num_queries: usize,
    dimension: usize,
    k: usize,
    l: u32,
    beam_width: usize,
    num_threads: usize,
    cache_nodes: usize,
    total_elapsed_ms: f64,
    qps: f64,
    results: Vec<QueryResultItem>,
}

/// Reads a matrix of f32 vectors from a DiskANN binary file.
fn load_bin_vectors(path: &Path) -> Result<(usize, usize, Vec<f32>)> {
    let mut file = BufReader::new(
        File::open(path).with_context(|| format!("Failed to open queries file: {}", path.display()))?,
    );
    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;

    let npoints = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
    let ndims = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;

    if npoints == 0 || ndims == 0 {
        bail!("Invalid empty dataset in {}: {} points, {} dims", path.display(), npoints, ndims);
    }

    let total_floats = npoints * ndims;
    let mut data = vec![0.0f32; total_floats];
    let slice_u8: &mut [u8] = bytemuck::cast_slice_mut(&mut data);
    file.read_exact(slice_u8)
        .with_context(|| format!("Failed to read vector payload from {}", path.display()))?;

    Ok((npoints, ndims, data))
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 1. Validate parameters
    if args.top_k == 0 {
        bail!("top_k must be strictly positive (got 0)");
    }
    if (args.search_l as usize) < args.top_k {
        bail!(
            "search_l ({}) must be >= top_k ({})",
            args.search_l,
            args.top_k
        );
    }

    let prefix_str = args.index_prefix.to_string_lossy().to_string();

    // 2. Validate index files existence
    let disk_index_path = get_disk_index_file(&prefix_str);
    let pq_pivot_path = get_pq_pivot_file(&prefix_str);
    let pq_data_path = get_compressed_pq_file(&prefix_str);

    for (p, desc) in [
        (&disk_index_path, "disk index file"),
        (&pq_pivot_path, "pq pivot file"),
        (&pq_data_path, "compressed pq file"),
    ] {
        if !Path::new(p).is_file() {
            bail!("Required index artifact {} not found at {}", desc, p);
        }
    }

    // 3. Load ID map if provided
    let local_to_global: Option<Arc<Vec<usize>>> = if let Some(map_path) = &args.id_map {
        let file = File::open(map_path)
            .with_context(|| format!("Failed to open id_map at {}", map_path.display()))?;
        let parsed: IdMappingFile = serde_json::from_reader(BufReader::new(file))
            .with_context(|| format!("Failed to parse id_map at {}", map_path.display()))?;
        Some(Arc::new(parsed.local_to_global))
    } else {
        None
    };

    // 4. Load query vectors
    let (npoints, ndims, flat_queries) = if let Some(queries_path) = &args.queries {
        load_bin_vectors(queries_path)?
    } else if let Some(raw_json) = &args.single_query_json {
        let parsed: Vec<f32> = serde_json::from_str(raw_json)
            .context("Failed to parse single_query_json as array of floats")?;
        let len = parsed.len();
        (1, len, parsed)
    } else {
        bail!("Either --queries or --single-query-json must be specified");
    };

    // Validate finite values
    for (i, v) in flat_queries.iter().enumerate() {
        if !v.is_finite() {
            bail!("Query vector contains non-finite value at float index {}: {}", i, v);
        }
    }

    // 5. Initialize DiskANN3 search engine
    let caching_strategy = if args.cache_nodes > 0 {
        CachingStrategy::StaticCacheWithBfsNodes(args.cache_nodes)
    } else {
        CachingStrategy::None
    };

    let index_reader = DiskIndexReader::new(pq_pivot_path, pq_data_path, &FileStorageProvider)
        .context("Failed to initialize DiskIndexReader")?;

    let vertex_provider_factory =
        DiskVertexProviderFactory::from_disk_index_path(disk_index_path, caching_strategy)
            .context("Failed to initialize DiskVertexProviderFactory")?;

    let searcher = Arc::new(DiskIndexSearcher::<AdHoc<f32>, _>::new(
        args.num_threads,
        usize::MAX,
        &index_reader,
        vertex_provider_factory,
        Metric::L2,
        None,
    ).context("Failed to create DiskIndexSearcher")?);

    // 6. Execute search
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.num_threads)
        .build()
        .context("Failed to create rayon thread pool")?;

    let start_all = Instant::now();

    let results: Vec<QueryResultItem> = pool.install(|| {
        (0..npoints)
            .into_par_iter()
            .map(|qi| {
                let q_slice = &flat_queries[qi * ndims..(qi + 1) * ndims];
                let t0 = Instant::now();

                let mode = SearchMode::Graph { filter: None };
                let search_res = searcher
                    .search(
                        q_slice,
                        args.top_k as u32,
                        args.search_l,
                        Some(args.beam_width),
                        mode,
                    )
                    .unwrap_or_else(|e| {
                        panic!("Search error for query {}: {:?}", qi, e);
                    });

                let micros = t0.elapsed().as_micros() as u64;

                let mut local_ids = Vec::with_capacity(search_res.results.len());
                let mut dists = Vec::with_capacity(search_res.results.len());

                for item in search_res.results.iter().take(args.top_k) {
                    local_ids.push(item.vertex_id);
                    dists.push(item.distance);
                }

                let global_ids = local_to_global.as_ref().map(|l2g| {
                    local_ids
                        .iter()
                        .map(|&lid| {
                            if (lid as usize) < l2g.len() {
                                l2g[lid as usize]
                            } else {
                                usize::MAX
                            }
                        })
                        .collect()
                });

                QueryResultItem {
                    query_idx: qi,
                    local_ids,
                    global_ids,
                    distances: dists,
                    duration_micros: micros,
                }
            })
            .collect()
    });

    let total_elapsed = start_all.elapsed();
    let total_elapsed_ms = total_elapsed.as_secs_f64() * 1000.0;
    let qps = (npoints as f64) / total_elapsed.as_secs_f64();

    let report = OutputReport {
        index_prefix: prefix_str,
        num_queries: npoints,
        dimension: ndims,
        k: args.top_k,
        l: args.search_l,
        beam_width: args.beam_width,
        num_threads: args.num_threads,
        cache_nodes: args.cache_nodes,
        total_elapsed_ms,
        qps,
        results,
    };

    if let Some(out_path) = &args.output {
        let file = File::create(out_path)
            .with_context(|| format!("Failed to create output file {}", out_path.display()))?;
        serde_json::to_writer_pretty(BufWriter::new(file), &report)?;
        println!(
            "Searched {} queries in {:.2}ms (QPS: {:.1}). Output saved to {}",
            npoints,
            total_elapsed_ms,
            qps,
            out_path.display()
        );
    } else {
        let serialized = serde_json::to_string_pretty(&report)?;
        println!("{}", serialized);
    }

    Ok(())
}
