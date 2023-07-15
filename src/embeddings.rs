use anyhow::Result;
use csv::Reader;
use indicatif::ProgressBar;
use mimalloc::MiMalloc;
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel;
use serde::Deserialize;
use std::{
    env,
    fs::{read_to_string, File},
    process,
    time::Instant,
};

mod utils;
use utils::load_model;

use semantic_search::hnsw_index::{api::AnnT, dist::DistL2, hnsw::Hnsw};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Debug, Clone)]
struct Config {
    dataset: String,
}
impl Config {
    fn new(args: &[String]) -> Result<Config, &'static str> {
        if args.len() < 1 {
            return Err("not enough arguments");
        }

        let dataset: String = args[0].to_string();

        Ok(Config { dataset })
    }
}

#[derive(Debug, Deserialize)]
pub struct Library {
    pub books: Vec<Book>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Book {
    pub title: String,

    pub author: String,

    pub summary: String,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    let config: Config = Config::new(&args).unwrap_or_else(|err: &str| {
        println!("Problem parsing arguments: {}", err);
        println!("Usage: embeddings dataset (news or book)");
        process::exit(1);
    });

    let start: Instant = Instant::now();
    let model: SentenceEmbeddingsModel = load_model();
    println!("load model : {:.3?}", start.elapsed());

    let start: Instant = Instant::now();

    let data: Vec<String>;
    if config.dataset == "book" {
        let json: String = read_to_string("data/books.json")?;
        let library: Library = serde_json::from_str(&json)?;

        data = library
            .books
            .iter()
            .map(|book: &Book| book.summary.clone())
            .collect();
    } else {
        let file: File = File::open("data/ag_news.csv")?;
        let mut reader = Reader::from_reader(file);

        data = reader
            .records()
            .map(|res| res.unwrap()[0].to_string())
            .collect();
    }
    println!("load data : {:.3?}", start.elapsed());

    let nb_elem: usize = data.len();
    let max_nb_connection: usize = 16;
    let ef_c: usize = 200;
    let nb_layer: usize = 16;
    let index: Hnsw<f32, DistL2> =
        Hnsw::<f32, DistL2>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistL2 {});

    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(nb_elem);

    let bs: usize = 128;
    let pb = ProgressBar::new((nb_elem / bs + 1) as u64);
    for chunk in data.chunks(bs) {
        let embeds: Vec<Vec<f32>> = model.encode(chunk).unwrap();
        embeddings.extend(embeds.into_iter());
        pb.inc(1);
    }
    pb.finish();

    println!("inference : {:.3?}", pb.elapsed());

    let embeddings_indices: Vec<(&Vec<f32>, usize)> =
        embeddings.iter().zip(0..embeddings.len()).collect();

    let start: Instant = Instant::now();
    index.parallel_insert(&embeddings_indices);
    println!("parallel insert : {:.3?}", start.elapsed());

    _ = index.file_dump(&config.dataset);

    Ok(())
}
