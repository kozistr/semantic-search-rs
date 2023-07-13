use std::{
    fs::{File, OpenOptions},
    io::BufReader,
    path::PathBuf,
};

use semantic_search::hnsw_index::{
    dist::DistL2,
    hnsw::Hnsw,
    hnswio::{load_description, load_hnsw, Description},
};

fn load_file(filename: &str) -> BufReader<File> {
    let path: PathBuf = PathBuf::from(filename.to_string());
    let res: File = OpenOptions::new().read(true).open(&path).unwrap();
    let reader: BufReader<File> = BufReader::new(res);
    reader
}

fn load_index() -> Hnsw<f32, DistL2> {
    println!("load index");
    let mut graph: BufReader<File> = load_file("index.hnsw.graph");
    let mut data: BufReader<File> = load_file("index.hnsw.data");

    let description: Description = load_description(&mut graph).unwrap();
    let index: Hnsw<f32, DistL2> = load_hnsw(&mut graph, &description, &mut data).unwrap();
    index
}

fn main() {
    let index: Hnsw<f32, DistL2> = load_index();
}
