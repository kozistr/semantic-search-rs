use hora::core::ann_index::{ANNIndex, SerializableIndex};
use hora::index::hnsw_idx::HNSWIndex;
use mimalloc::MiMalloc;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType::AllMiniLmL12V2,
};
use std::time::Instant;

use crate::ss::{Features, Index, PredictRequest, PredictResponse};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

thread_local! {
    pub static MODEL: SentenceEmbeddingsModel = load_model();
    pub static INDEX: HNSWIndex<f32, usize> = load_index();
}

fn load_model() -> SentenceEmbeddingsModel {
    println!("load model");
    SentenceEmbeddingsBuilder::remote(AllMiniLmL12V2)
        .create_model()
        .unwrap()
}

fn load_index() -> HNSWIndex<f32, usize> {
    println!("load index");
    HNSWIndex::<f32, usize>::load("index.hora").unwrap()
}

#[allow(dead_code)]
pub fn preprocess(features: &Vec<Features>) -> (Vec<Vec<String>>, Vec<Vec<usize>>) {
    let query: Vec<Vec<String>> = features
        .iter()
        .map(|f: &Features| vec![f.query.clone()])
        .collect();

    let k: Vec<Vec<usize>> = features
        .iter()
        .map(|f: &Features| vec![f.k as usize])
        .collect();

    (query, k)
}

pub fn search(request: PredictRequest) -> PredictResponse {
    // let (query, k) = preprocess(&request.features);
    let feature: &Features = &request.features.first().clone().unwrap();
    let query = feature.query.clone();
    let k: usize = feature.k.clone() as usize;

    let start: Instant = Instant::now();
    let query_embeddings: Vec<Vec<f32>> =
        MODEL.with(|model: &SentenceEmbeddingsModel| model.encode(&[query]).unwrap());
    let model_latency: u64 = start.elapsed().as_nanos() as u64;

    let start: Instant = Instant::now();
    let neighbor_index: Vec<usize> =
        INDEX.with(|index: &HNSWIndex<f32, usize>| index.search(&query_embeddings[0], k));
    let search_latency: u64 = start.elapsed().as_nanos() as u64;

    PredictResponse {
        indices: neighbor_index
            .iter()
            .map(|index: &usize| Index {
                index: *index as i32,
            })
            .collect(),
        model_latency,
        search_latency,
    }
}
