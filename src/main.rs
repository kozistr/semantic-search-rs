use csv;
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel;
use std::fs::File;

use semantic_search::{
    hnsw_index::{
        dist::DistL2,
        hnsw::{Hnsw, Neighbour},
    },
    utils::{load_index, load_model},
};

fn main() {
    let model: SentenceEmbeddingsModel = load_model();
    let index: Hnsw<f32, DistL2> = load_index("news");

    let file: File = File::open("data/ag_news.csv").unwrap();
    let mut reader = csv::Reader::from_reader(file);

    let data: Vec<String> = reader
        .records()
        .map(|res| res.unwrap()[0].to_string())
        .collect();

    let query: String = "Asia shares drift lower as investors factor in Fed rate hike.".to_string();

    let query_embedding: Vec<Vec<f32>> = model.encode(&[query]).unwrap();

    let neighbors: Vec<Neighbour> = index.search(&query_embedding[0], 10, 30);

    for (k, neighbor) in neighbors.iter().enumerate() {
        println!(
            "top {} | id : {}, dist : {}",
            k + 1,
            neighbor.d_id,
            neighbor.distance,
        );
        println!("{}", data[neighbor.d_id]);
    }
}
