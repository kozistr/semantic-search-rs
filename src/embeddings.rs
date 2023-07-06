use anyhow::Result;
// use hora::{
//     core::{ann_index::ANNIndex, ann_index::SerializableIndex, metrics::Metric::Euclidean},
//     index::{hnsw_idx::HNSWIndex, hnsw_params::HNSWParams},
// };
// use hnsw_rs::prelude::{AnnT, DistL2, Hnsw};
use mimalloc::MiMalloc;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType::AllMiniLmL12V2,
};
use serde::Deserialize;
use std::fs;

mod hnsw_index;
use crate::hnsw_index::{api::AnnT, dist::DistL2, hnsw::Hnsw};

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

    // hora
    // let mut index: HNSWIndex<f32, usize> =
    //     HNSWIndex::<f32, usize>::new(384, &HNSWParams::<f32>::default());

    // library.books.iter().enumerate().for_each(|(idx, book)| {
    //     if let Ok(embedding) = model.encode(&[book.summary.clone()]) {
    //         index.add(&embedding[0], idx).unwrap();
    //     }
    // });

    // index.build(Euclidean)?;

    // index.dump("index.hora")?;

    // hnswlib
    let nb_elem: usize = library.books.len();
    let max_nb_connection: usize = 16;
    let ef_c: usize = 200;
    let nb_layer: usize = 16;
    let index: Hnsw<f32, DistL2> =
        Hnsw::<f32, DistL2>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistL2 {});

    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(nb_elem);
    library.books.iter().for_each(|book: &Book| {
        if let Ok(embedding) = model.encode(&[book.summary.clone()]) {
            embeddings.push(embedding[0].to_vec());
        }
    });
    let embeddings_indices: Vec<(&Vec<f32>, usize)> =
        embeddings.iter().zip(0..embeddings.len()).collect();

    index.parallel_insert(&embeddings_indices);

    _ = index.file_dump(&"index".to_string());

    Ok(())
}
