use hora::core::ann_index::SerializableIndex;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType::AllMiniLmL12V2,
};
use serde::Deserialize;
use std::{error::Error, fs, path::Path, time::Instant};

use hora::core::{ann_index::ANNIndex, metrics::Metric::Euclidean};
use hora::index::{hnsw_idx::HNSWIndex, hnsw_params::HNSWParams};

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

impl Book {
    fn to_default_embedded(self) -> EmbeddedBook {
        EmbeddedBook {
            title: Some(self.title),
            author: Some(self.author),
            summary: Some(self.summary),
            embeddings: to_array(&vec![0f32; 384]),
        }
    }

    fn to_embedded(self, embeddings: [f32; 384]) -> EmbeddedBook {
        EmbeddedBook {
            title: Some(self.title),
            author: Some(self.author),
            summary: Some(self.summary),
            embeddings,
        }
    }
}
#[derive(Debug, Clone)]
pub struct EmbeddedBook {
    pub title: Option<String>,

    pub author: Option<String>,

    pub summary: Option<String>,

    pub embeddings: [f32; 384],
}

fn to_array(barry: &[f32]) -> [f32; 384] {
    barry.try_into().expect("slice with incorrect length")
}

fn main() -> Result<(), Box<dyn Error>> {
    let now: Instant = Instant::now();
    let model: SentenceEmbeddingsModel =
        SentenceEmbeddingsBuilder::remote(AllMiniLmL12V2).create_model()?;
    println!("load model : {:?}", now.elapsed());

    let now: Instant = Instant::now();
    let json: String = fs::read_to_string("data/books.json")?;
    let library: Library = serde_json::from_str(&json)?;
    println!("load data : {:?}", now.elapsed());

    let ret: (HNSWIndex<f32, usize>, Vec<EmbeddedBook>);
    if Path::new("index.hora").exists() == false {
        println!("[-] there's no index file.");

        // let mut summaries: Vec<String> = Vec::new();
        // for book in library.books.clone() {
        //     summaries.push(book.summary);
        // }

        // let now: Instant = Instant::now();
        // let embeddings: Vec<Vec<f32>> = model.encode(&summaries)?;
        // println!(
        //     "batch inference ({:?} documents) : {:?}",
        //     summaries.len(),
        //     now.elapsed()
        // );

        let now: Instant = Instant::now();
        let mut embeddings: Vec<[f32; 384]> = Vec::new();
        for book in library.books.iter() {
            let embedding: Vec<Vec<f32>> = model.encode(&[book.clone().summary])?;
            embeddings.push(to_array(&embedding[0]));
        }
        println!(
            "inference ({:?} documents) : {:?}",
            embeddings.len(),
            now.elapsed()
        );

        let mut embeddedbooks: Vec<EmbeddedBook> = Vec::new();
        for (book, embedding) in library.books.iter().zip(embeddings.iter()) {
            embeddedbooks.push(book.clone().to_embedded(to_array(embedding)));
        }

        let now: Instant = Instant::now();
        let mut index: HNSWIndex<f32, usize> =
            HNSWIndex::<f32, usize>::new(384, &HNSWParams::<f32>::default());
        for (i, embedding) in embeddings.iter().enumerate() {
            index.add(embedding, i).unwrap();
        }
        index.build(Euclidean).unwrap();
        index.dump("index.hora").unwrap();
        println!("build index : {:?}", now.elapsed());

        ret = (index, embeddedbooks);
    } else {
        println!("[+] there's an index file.");

        let mut embeddedbooks: Vec<EmbeddedBook> = Vec::new();
        for book in library.books {
            embeddedbooks.push(book.clone().to_default_embedded());
        }

        let index = HNSWIndex::<f32, usize>::load("index.hora").unwrap();

        ret = (index, embeddedbooks);
    };

    let (index, embeddedbooks) = ret;

    let query: &str = "The story about prep school";
    println!("query : {}", query);

    let query_embeddings: Vec<Vec<f32>> = model.encode(&[query])?;

    let now: Instant = Instant::now();
    let neighbor_index: Vec<usize> = index.search(&query_embeddings[0], 10);
    println!("search speed : {:?}", now.elapsed());

    for (k, idx) in neighbor_index.iter().enumerate() {
        let book: EmbeddedBook = embeddedbooks[*idx].clone();
        println!("top {:?}, title : {:?}", k + 1, book.title);
    }

    Ok(())
}
