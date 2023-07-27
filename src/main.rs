use std::time::Instant;
use std::{env, process};

use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel;
use semantic_search::hnsw_index::dist::DistDot;
use semantic_search::hnsw_index::hnsw::{quantize, Hnsw, Neighbour};
use semantic_search::utils::{load_data, load_index, load_model, load_quantize_index, log_stats};

static BENCH_SIZE: usize = 2000;
static K: usize = 10;

#[allow(dead_code)]
fn find_documents(query: String, do_quantize: bool) {
    println!("query : {:?}", query);
    println!("do quantize : {:?}", do_quantize);

    let data: Vec<String> = load_data();
    let model: SentenceEmbeddingsModel = load_model();

    let query_embedding: Vec<Vec<f32>> = model.encode(&[query]).unwrap();

    let neighbors: Vec<Neighbour> = if !do_quantize {
        let index: Hnsw<f32, DistDot> = load_index("news");

        index.search(&query_embedding[0], K, 30)
    } else {
        let index: Hnsw<i8, DistDot> = load_quantize_index("news");

        let query_embedding: Vec<i8> = quantize(&query_embedding[0]);

        index.search(query_embedding.as_slice(), K, 30)
    };

    for (k, neighbor) in neighbors.iter().enumerate() {
        println!("top {} | id : {}, dist : {}", k + 1, neighbor.d_id, neighbor.distance);
        println!("{}", data[neighbor.d_id]);
    }
}

#[allow(dead_code)]
fn bench_search() {
    // let index: Hnsw<f32, DistDot> = load_index("news");
    let index: Hnsw<i8, DistDot> = load_quantize_index("news");
    let query_embedding: Vec<i8> = quantize(&query_embedding[0]);

    for bs in [1024, 2048, 4096, 8192] {
        let query_embeddings: Vec<Vec<i8>> = vec![query_embedding.clone(); bs];
        // let query_embeddings: Vec<Vec<f32>> = vec![query_embedding[0].clone(); bs];
        let mut search_lat: Vec<u64> = vec![0u64; BENCH_SIZE];

        (0..BENCH_SIZE).for_each(|i: usize| {
            let start: Instant = Instant::now();
            _ = index.parallel_search(&query_embeddings, K, 30);
            search_lat[i] = start.elapsed().as_nanos() as u64;
        });

        log_stats("search", BENCH_SIZE, bs, &search_lat);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        println!("Usage: main query [full or quantize]");
        process::exit(1);
    }

    // let query: String = args[1].clone();
    // let do_quantize: bool = args[2] == "quantize";
    // find_documents(query, do_quantize);

    bench_search();
}
