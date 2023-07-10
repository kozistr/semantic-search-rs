use hora::core::ann_index::{ANNIndex, SerializableIndex};
use hora::index::hnsw_idx::HNSWIndex;
use mimalloc::MiMalloc;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType::AllMiniLmL12V2,
};

use crate::ss::{Features, Index, PredictRequest, PredictResponse};

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
    HNSWIndex::<f32, usize>::load("index.hora").unwrap()
}

pub fn search(request: PredictRequest) -> PredictResponse {
    let feature: &Features = &request.features.first().clone().unwrap();
    let query = feature.query.clone();
    let k: usize = feature.k.clone() as usize;

    // let now: Instant = Instant::now();
    let query_embeddings: Vec<Vec<f32>> =
        MODEL.with(|model: &SentenceEmbeddingsModel| model.encode(&[query]).unwrap());
    // println!("embedding : {:?}", now.elapsed());

    // let now: Instant = Instant::now();
    let neighbor_index: Vec<usize> =
        INDEX.with(|index: &HNSWIndex<f32, usize>| index.search(&query_embeddings[0], k));
    // println!("search : {:?}", now.elapsed());

    PredictResponse {
        indices: neighbor_index
            .iter()
            .map(|index| Index {
                index: *index as i32,
            })
            .collect(),
    }
}
