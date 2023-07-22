use std::time::Instant;

use mimalloc::MiMalloc;
use rayon::prelude::*;
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel;

use crate::hnsw_index::dist::DistDot;
use crate::hnsw_index::hnsw::{quantize, Hnsw, Neighbour};
use crate::ss::{Features, Index, PredictRequest, PredictResponse};
use crate::utils::{load_model, load_quantize_index};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

thread_local! {
    pub static MODEL: SentenceEmbeddingsModel = load_model();
    // pub static INDEX: Hnsw<f32, DistDot> = load_index("news");
    pub static INDEX: Hnsw<i8, DistDot> = load_quantize_index("news");
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

    let query_embeddings: Vec<Vec<i8>> = query_embeddings
        .par_iter()
        .map(|embedding: &Vec<f32>| quantize(embedding))
        .collect();

    let start: Instant = Instant::now();
    let neighbor_index: Vec<Vec<Neighbour>> =
        INDEX.with(|index: &Hnsw<i8, DistDot>| index.parallel_search(&query_embeddings, k, 30));
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
