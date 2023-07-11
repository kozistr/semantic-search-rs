use anyhow::Result;
use hora::{
    core::{ann_index::ANNIndex, ann_index::SerializableIndex, metrics::Metric::Euclidean},
    index::{hnsw_idx::HNSWIndex, hnsw_params::HNSWParams},
};
use mimalloc::MiMalloc;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType::AllMiniLmL12V2,
};
use serde::Deserialize;
use std::fs;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

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
#[derive(Debug, Clone)]
pub struct EmbeddedBook {
    pub title: Option<String>,

    pub author: Option<String>,

    pub summary: Option<String>,

    pub embeddings: [f32; 384],
}

fn main() -> Result<()> {
    let model: SentenceEmbeddingsModel =
        SentenceEmbeddingsBuilder::remote(AllMiniLmL12V2).create_model()?;

    let json: String = fs::read_to_string("data/books.json")?;
    let library: Library = serde_json::from_str(&json)?;

    let mut index: HNSWIndex<f32, usize> =
        HNSWIndex::<f32, usize>::new(384, &HNSWParams::<f32>::default());

    library.books.iter().enumerate().for_each(|(idx, book)| {
        if let Ok(embedding) = model.encode(&[book.summary.clone()]) {
            index.add(&embedding[0], idx).unwrap();
        }
    });

    index.build(Euclidean)?;

    index.dump("index.hora")?;

    Ok(())
}
