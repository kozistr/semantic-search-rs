use std::time::Instant;

use anyhow::Result;
#[cfg(feature = "progress")]
use indicatif::ProgressBar;
use rayon::prelude::*;
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel;
use semantic_search::hnsw_index::api::AnnT;
use semantic_search::hnsw_index::dist::DistDot;
use semantic_search::hnsw_index::hnsw::{quantize, Hnsw};
use semantic_search::utils::{load_data, load_model};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    let do_quantize: bool = if args.len() < 2 { false } else { true };
    println!("do quantize (f32 to i8) : {:?}", do_quantize);

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

    let index: Hnsw<_, DistDot> = if !do_quantize {
        Hnsw::<f32, DistDot>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistDot {})
    } else {
        Hnsw::<i8, DistDot>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistDot {})
    };

    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(nb_elem);

    let bs: usize = 128;

    let pb;
    #[cfg(feature = "progress")]
    {
        pb = ProgressBar::new((nb_elem / bs + 1) as u64);
    }
    #[cfg(not(feature = "progress"))]
    {
        pb = Instant::now();
    }

    for chunk in data.chunks(bs) {
        let embeds: Vec<Vec<f32>> = model.encode(chunk).unwrap();
        embeddings.extend(embeds.into_iter());
        #[cfg(feature = "progress")]
        {
            pb.inc(1);
        }
    }
    #[cfg(feature = "progress")]
    {
        pb.finish();
    }

    println!("inference : {:.3?}", pb.elapsed());

    let embeddings: Vec<Vec<_>> = if do_quantize {
        let start: Instant = Instant::now();
        let embeddings: Vec<Vec<i8>> = embeddings
            .par_iter()
            .map(|embedding: &Vec<f32>| quantize(embedding))
            .collect();
        println!("quantize : {:.3?}", start.elapsed());
        embeddings
    } else {
        embeddings
    };

    let embeddings_indices: Vec<(&Vec<i8>, usize)> =
        embeddings.iter().zip(0..embeddings.len()).collect();

    let start: Instant = Instant::now();
    index.parallel_insert(&embeddings_indices);
    println!("parallel insert : {:.3?}", start.elapsed());

    _ = index.file_dump(&"news_q".to_string());

    Ok(())
}
