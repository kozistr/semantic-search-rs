use std::{
    fs::{File, OpenOptions},
    io::BufReader,
    path::{Path, PathBuf},
};

use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType::AllMiniLmL12V2,
};

use crate::hnsw_index::{
    dist::DistCosine,
    hnsw::Hnsw,
    hnswio::{load_description, load_hnsw, Description},
};

pub fn load_model() -> SentenceEmbeddingsModel {
    let model: SentenceEmbeddingsModel = if Path::new("models").is_dir() {
        println!("load model from local");
        SentenceEmbeddingsBuilder::local("models")
            .create_model()
            .unwrap()
    } else {
        println!("load model from remote");
        SentenceEmbeddingsBuilder::remote(AllMiniLmL12V2)
            .create_model()
            .unwrap()
    };
    model
}

fn load_file(filename: &String) -> BufReader<File> {
    let path: PathBuf = PathBuf::from(filename);
    let res: File = OpenOptions::new().read(true).open(&path).unwrap();
    let reader: BufReader<File> = BufReader::new(res);
    reader
}

pub fn load_index(dataset: &str) -> Hnsw<f32, DistCosine> {
    println!("load index");
    let mut graph: BufReader<File> = load_file(&format!("{}.hnsw.graph", dataset));
    // todo: offload to the disk (memmmap) to save the memmory
    let mut data: BufReader<File> = load_file(&format!("{}.hnsw.data", dataset));

    let description: Description = load_description(&mut graph).unwrap();
    let index: Hnsw<f32, DistCosine> = load_hnsw(&mut graph, &description, &mut data).unwrap();
    index
}
