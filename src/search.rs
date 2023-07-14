// use hora::{
//     core::ann_index::{ANNIndex, SerializableIndex},
//     index::hnsw_idx::HNSWIndex,
// };
use mimalloc::MiMalloc;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType::AllMiniLmL12V2,
};
use std::{
    fs::{File, OpenOptions},
    io::BufReader,
    path::PathBuf,
    time::Instant,
};

use semantic_search::hnsw_index::{
    dist::DistL2,
    hnsw::{Hnsw, Neighbour},
    hnswio::{load_description, load_hnsw_with_dist, Description},
};

use crate::ss::{Features, Index, PredictRequest, PredictResponse};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

thread_local! {
    pub static MODEL: SentenceEmbeddingsModel = load_model();
    pub static INDEX: Hnsw<f32, DistL2> = load_index();
}

fn load_model() -> SentenceEmbeddingsModel {
    println!("load model");
    // SentenceEmbeddingsBuilder::remote(AllMiniLmL12V2)
    //     .create_model()
    //     .unwrap()
    SentenceEmbeddingsBuilder::local("models")
        .create_model()
        .unwrap()
}

fn load_file(filename: &str) -> BufReader<File> {
    let path: PathBuf = PathBuf::from(filename.to_string());
    let res: File = OpenOptions::new().read(true).open(&path).unwrap();
    let reader: BufReader<File> = BufReader::new(res);
    reader
}

fn load_index() -> Hnsw<f32, DistL2> {
    println!("load index");
    let mut graph: BufReader<File> = load_file("news.hnsw.graph");
    // todo: offload to the disk (memmmap) to save the memmory
    let mut data: BufReader<File> = load_file("news.hnsw.data");

    let description: Description = load_description(&mut graph).unwrap();
    let index: Hnsw<f32, DistL2> =
        load_hnsw_with_dist(&mut graph, &description, DistL2 {}, &mut data).unwrap();
    index
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
    // let neighbor_index: Vec<usize> =
    //     INDEX.with(|index: &HNSWIndex<f32, usize>| index.search(&query_embeddings[0], k));
    let neighbor_index: Vec<Vec<Neighbour>> =
        INDEX.with(|index: &Hnsw<f32, DistL2>| index.parallel_search(&query_embeddings, k, 30));
    let search_latency: u64 = start.elapsed().as_nanos() as u64;

    PredictResponse {
        // indices: neighbor_index
        //     .iter()
        //     .map(|index: &usize| Index {
        //         index: *index as i32,
        //     })
        //     .collect(),
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
