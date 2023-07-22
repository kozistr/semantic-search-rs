use std::env;
use std::time::Instant;

use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel;
use semantic_search::hnsw_index::dist::DistDot;
use semantic_search::hnsw_index::hnsw::{Hnsw, Neighbour};
use semantic_search::utils::{load_data, load_index, load_model};

fn main() {
    let args: Vec<String> = env::args().collect();

    let query: String = if args.len() < 2 {
        "Asia shares drift lower as investors factor in Fed rate hike.".to_string()
    } else {
        args[1].clone()
    };
    println!("query : {:?}", query);

    let model: SentenceEmbeddingsModel = load_model();
    let index: Hnsw<f32, DistDot> = load_index("news");
    let data: Vec<String> = load_data();

    let query_embedding: Vec<Vec<f32>> = model.encode(&[query]).unwrap();

    let neighbors: Vec<Neighbour> = index.search(&query_embedding[0], 10, 30);

    for (k, neighbor) in neighbors.iter().enumerate() {
        println!("top {} | id : {}, dist : {}", k + 1, neighbor.d_id, neighbor.distance);
        println!("{}", data[neighbor.d_id]);
    }
}
