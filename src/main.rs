use std::{env, process};

use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel;
use semantic_search::hnsw_index::dist::DistDot;
use semantic_search::hnsw_index::hnsw::{quantize, Hnsw, Neighbour};
use semantic_search::utils::{load_data, load_index, load_model, load_quantize_index};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        println!("Usage: main query [full or quantize]");
        process::exit(1);
    }

    let query: String = args[1].clone();
    let do_quantize: bool = if args[2] == "quantize" { true } else { false };

    println!("query : {:?}", query);
    println!("do quantize : {:?}", do_quantize);

    let data: Vec<String> = load_data();
    let model: SentenceEmbeddingsModel = load_model();

    let query_embedding: Vec<Vec<f32>> = model.encode(&[query]).unwrap();

    let neighbors: Vec<Neighbour> = if !do_quantize {
        let index: Hnsw<f32, DistDot> = load_index("news");

        index.search(&query_embedding[0], 10, 30)
    } else {
        let index: Hnsw<i8, DistDot> = load_quantize_index("news");

        let query_embedding: Vec<i8> = quantize(&query_embedding[0]);

        index.search(query_embedding.as_slice(), 10, 30)
    };

    for (k, neighbor) in neighbors.iter().enumerate() {
        println!("top {} | id : {}, dist : {}", k + 1, neighbor.d_id, neighbor.distance);
        println!("{}", data[neighbor.d_id]);
    }
}
