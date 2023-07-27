use std::fs::{File, OpenOptions};
use std::io::BufReader;
use std::path::{Path, PathBuf};

use csv;
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModelType::AllMiniLmL12V2;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel,
};

use crate::hnsw_index::dist::DistDot;
use crate::hnsw_index::hnsw::Hnsw;
use crate::hnsw_index::hnswio::{load_description, load_hnsw, Description};

pub fn load_data() -> Vec<String> {
    let file: File = File::open("./data/ag_news.csv").unwrap();
    let mut reader: csv::Reader<File> = csv::Reader::from_reader(file);

    let data: Vec<String> = reader
        .records()
        .map(|res: Result<csv::StringRecord, csv::Error>| res.unwrap()[0].to_string())
        .collect();
    data
}

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
    let res: File = OpenOptions::new().read(true).open(path).unwrap();
    let reader: BufReader<File> = BufReader::new(res);
    reader
}

#[allow(unused)]
pub fn load_index(dataset: &str) -> Hnsw<f32, DistDot> {
    println!("load index");

    let index: Hnsw<f32, DistDot> = {
        let mut graph: BufReader<File> = load_file(&format!("{}.hnsw.graph", dataset));
        // todo: offload to the disk (memmmap) to save the memmory
        let mut data: BufReader<File> = load_file(&format!("{}.hnsw.data", dataset));

        let description: Description = load_description(&mut graph).unwrap();

        let mut index: Hnsw<f32, DistDot> = load_hnsw(&mut graph, &description, &mut data).unwrap();
        index.set_searching_mode(true);

        index
    };

    index
}

#[allow(unused)]
pub fn load_quantize_index(dataset: &str) -> Hnsw<i8, DistDot> {
    println!("load quantize index");

    let index: Hnsw<i8, DistDot> = {
        let mut graph: BufReader<File> = load_file(&format!("{}_q.hnsw.graph", dataset));
        // todo: offload to the disk (memmmap) to save the memmory
        let mut data: BufReader<File> = load_file(&format!("{}_q.hnsw.data", dataset));

        let description: Description = load_description(&mut graph).unwrap();

        let mut index: Hnsw<i8, DistDot> = load_hnsw(&mut graph, &description, &mut data).unwrap();
        index.set_searching_mode(true);

        index
    };

    index
}

fn percentiles(ps: &[f32], lats: &Vec<u64>) -> Vec<(f32, u64)> {
    ps.iter()
        .map(|p: &f32| (*p, lats[((lats.len() as f32) * p) as usize]))
        .collect()
}

pub fn log_stats(description: &str, i: usize, bs: usize, latencies: &Vec<u64>) {
    let mut lats: Vec<u64> = latencies.to_owned();
    lats.sort_unstable();

    let mean: f64 = (lats.clone().iter().sum::<u64>() / i as u64) as f64 * 1e-6;
    let max: f64 = *lats.clone().last().unwrap() as f64 * 1e-6;

    let ps: Vec<String> = percentiles(&[0.5, 0.95, 0.99, 0.999], &lats)
        .iter()
        .map(|(p, x)| format!("p{:2.1}={:1.3} ms", 100.0 * p, *x as f64 * 1e-6))
        .collect();

    println!(
        "{} latency : {} mean={:1.3} ms {} max={:1.3} ms QPS={:?}",
        description,
        i,
        mean,
        ps.join(" "),
        max,
        (1000.0 * (bs as f64 / mean)) as i32,
    );
}
