use std::time::Instant;

use anyhow::Result;
#[cfg(feature = "embedding")]
use indicatif::ProgressBar;
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel;
use semantic_search::hnsw_index::api::AnnT;
use semantic_search::hnsw_index::dist::DistDot;
use semantic_search::hnsw_index::hnsw::Hnsw;
use semantic_search::utils::{load_data, load_model};

fn main() -> Result<()> {
    let start: Instant = Instant::now();
    let model: SentenceEmbeddingsModel = load_model();
    println!("load model : {:.3?}", start.elapsed());

    let start: Instant = Instant::now();
    let data: Vec<String> = load_data();
    println!("load data : {:.3?}", start.elapsed());

    let nb_elem: usize = data.len();
    let max_nb_connection: usize = 16;
    let ef_c: usize = 200;
    let nb_layer: usize = 16;
    let index: Hnsw<f32, DistDot> =
        Hnsw::<f32, DistDot>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistDot {});

    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(nb_elem);

    let bs: usize = 128;

    let pb;
    #[cfg(feature = "embedding")]
    {
        pb = ProgressBar::new((nb_elem / bs + 1) as u64);
    }
    #[cfg(not(feature = "embedding"))]
    {
        pb = Instant::now();
    }

    for chunk in data.chunks(bs) {
        let embeds: Vec<Vec<f32>> = model.encode(chunk).unwrap();
        embeddings.extend(embeds.into_iter());
        #[cfg(feature = "embedding")]
        {
            pb.inc(1);
        }
    }
    #[cfg(feature = "embedding")]
    {
        pb.finish();
    }

    println!("inference : {:.3?}", pb.elapsed());

    let embeddings_indices: Vec<(&Vec<f32>, usize)> =
        embeddings.iter().zip(0..embeddings.len()).collect();

    let start: Instant = Instant::now();
    index.parallel_insert(&embeddings_indices);
    println!("parallel insert : {:.3?}", start.elapsed());

    _ = index.file_dump(&"news".to_string());

    Ok(())
}
