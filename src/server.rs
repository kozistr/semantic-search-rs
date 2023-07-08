use std::fs;

use anyhow::Ok;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use serde::Deserialize;

use hora::core::{
    ann_index::ANNIndex,
    metrics::Metric::Euclidean,
};
use hora::index::{
    hnsw_idx::HNSWIndex,
    hnsw_params::HNSWParams,
};

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
    fn to_embedded(self, embeddings: [f32; 384]) -> EmbeddedBook {
        EmbeddedBook {
            title: Some(self.title),
            author: Some(self.author),
            summary: Some(self.summary),
            embeddings,
        }
    }
}
#[derive(Debug)]
pub struct EmbeddedBook {
    pub title: Option<String>,

    pub author: Option<String>,

    pub summary: Option<String>,

    pub embeddings: [f32; 384],
}

impl EmbeddedBook {
    fn topic(embeddings: [f32; 384]) -> Self {
        Self {
            title: None,
            author: None,
            summary: None,
            embeddings,
        }
    }
}


fn main() -> anyhow::Result<()> {
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
        .create_model()?;

    let json = fs::read_to_string("data/books.json")?;
    let library: Library = serde_json::from_str(&json)?;

    let mut sentences = Vec::new();
    for book in library.books.clone() {
        sentences.push(book.summary);
    }

    // batch inference
    let embeddings = model.encode(&sentences)?;
    let mut embeddedbooks = Vec::new();
    for it in library.books.iter().zip(embeddings.iter()) {
        let (book, embedding) = it;
        embeddedbooks.push(book.clone().to_embedded(to_array(embedding)));
    }

    // init index
    let mut index = HNSWIndex::<f32, usize>::new(384, &HNSWParams::<f32>::default());
    for (i, sample) in embeddings.iter().enumerate() {
        index.add(sample, i).unwrap();
    }
    index.build(Euclidean).unwrap();

    let query = "What Gatsby does?";
    println!("Querying: {}", query);

    let query_embeddings = model.encode(&[query])?;
    // let query_embedding = to_array(query_embeddings[0].as_slice());
    // let query_topic = EmbeddedBook::topic(query_embedding);

    let neighbor_index = index.search(&query_embeddings[0], 5);
    println!("neighbors : {:?}", neighbor_index);

    for (k, idx) in neighbor_index.iter().enumerate() {
        let book = embeddedbooks[*idx];
        println!("top {:?}, title : {:?}", k + 1, book.title);
    }

    Ok(())
}

fn to_array(barry: &[f32]) -> [f32; 384] {
    barry.try_into().expect("slice with incorrect length")
}
