use mimalloc::MiMalloc;
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel;
use std::time::Instant;

use crate::{
    hnsw_index::{
        dist::DistL2,
        hnsw::{Hnsw, Neighbour},
    },
    ss::{Features, Index, PredictRequest, PredictResponse},
    utils::{load_index, load_model},
};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

thread_local! {
    pub static MODEL: SentenceEmbeddingsModel = load_model();
    pub static INDEX: Hnsw<f32, DistL2> = load_index("news");
}

pub fn preprocess(request: &PredictRequest) -> (Vec<String>, usize) {
    let query: Vec<String> = request
        .features
        .iter()
        .map(|f: &Features| f.query.clone() as String)
        .collect();

    let k: usize = request.k.clone() as usize;

    (query, k)
}

pub fn search(request: PredictRequest) -> PredictResponse {
    let (query, k) = preprocess(&request);

    let start: Instant = Instant::now();
    let query_embeddings: Vec<Vec<f32>> =
        MODEL.with(|model: &SentenceEmbeddingsModel| model.encode(&query).unwrap());
    let model_latency: u64 = start.elapsed().as_nanos() as u64;

    let start: Instant = Instant::now();
    let neighbor_index: Vec<Vec<Neighbour>> =
        INDEX.with(|index: &Hnsw<f32, DistL2>| index.parallel_search(&query_embeddings, k, 30));
    let search_latency: u64 = start.elapsed().as_nanos() as u64;

    PredictResponse {
        indices: neighbor_index
            .iter()
            .map(|indices: &Vec<Neighbour>| Index {
                index: indices
                    .iter()
                    .map(|idx: &Neighbour| idx.d_id as i32)
                    .collect(),
            })
            .collect(),
        model_latency,
        search_latency,
    }
}
