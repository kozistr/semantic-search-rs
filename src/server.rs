use std::fs;

use kd_tree::KdPoint;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use serde::Deserialize;

use std::time::Instant;

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

impl KdPoint for EmbeddedBook {
    type Scalar = f32;
    type Dim = typenum::U2;
    fn at(&self, k: usize) -> Self::Scalar {
        self.embeddings[k]
    }
}

fn main() -> anyhow::Result<()> {
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
        .create_model()?;

    let json = fs::read_to_string("data/books.json")?;
    let library: Library = serde_json::from_str(&json)?;

    let now = Instant::now();
    let mut embeddedbooks = Vec::new();
    for book in library.books.clone() {
        let embeddings = model.encode(&[book.clone().summary])?;

        embeddedbooks.push(book.to_embedded(to_array(embeddings[0].as_slice())));
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    let query = "asdf asdf asdf asdf asdf asdf asdf asdf";
    println!("Querying: {}", query);

    let query_embeddings = model.encode(&[query])?;
    let query_embedding = to_array(query_embeddings[0].as_slice());
    let query_topic = EmbeddedBook::topic(query_embedding);

    let kdtree = kd_tree::KdSlice::sort_by(&mut embeddedbooks, |item1, item2, k| {
        item1.embeddings[k]
            .partial_cmp(&item2.embeddings[k])
            .unwrap()
    });

    let nearests = kdtree.nearests(&query_topic, 3);
    for nearest in nearests {
        println!("nearest: {:?}", nearest.item.title);
        println!("distance: {:?}", nearest.squared_distance);
    }

    Ok(())
}

fn to_array(barry: &[f32]) -> [f32; 384] {
    barry.try_into().expect("slice with incorrect length")
}
