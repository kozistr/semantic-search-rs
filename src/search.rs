use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType::AllMiniLmL12V2,
};
use std::{error::Error, fs};

use hora::index::hnsw_idx::HNSWIndex;

use mimalloc::MiMalloc;

use crate::search::{Features, Index, PredictRequest, PredictResponse};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

thread_local! {
    pub static MODEL: SentenceEmbeddingsModel = load_model();
    pub static INDEX: HNSWIndex<f32, usize> = load_index();
}

fn load_model() -> SentenceEmbeddingsModel {
    SentenceEmbeddingsBuilder::remote(AllMiniLmL12V2)
        .create_model()
        .unwrap()
}

fn load_index() -> HNSWIndex<f32, usize> {
    SHNSWIndex::<f32, usize>::load("index.hora").unwrap()
}

pub fn search(request: PredictRequest) -> PredictResponse {
    let query: String = &request.query;
    let k: i32 = &request.k;

    // let now: Instant = Instant::now();
    let query_embeddings: Vec<Vec<f32>> = MODEL.encode(&[query])?;
    // println!("embedding : {:?}", now.elapsed());

    // let now: Instant = Instant::now();
    let neighbor_index: Vec<usize> = INDEX.search(&query_embeddings[0], k);
    // println!("search : {:?}", now.elapsed());

    PredictResponse {
        indices: neighbor_index
            .iter()
            .map(|score| Prediction {
                score: *score as i32,
            })
            .collect(),
    }
}
