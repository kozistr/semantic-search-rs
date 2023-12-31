use std::env;
use std::time::Instant;

use anyhow::Result;
#[cfg(feature = "progress")]
use indicatif::ProgressBar;
use rayon::prelude::*;
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel;
use semantic_search::hnsw_index::api::AnnT;
use semantic_search::hnsw_index::dist::{DistDot, DistHamming};
use semantic_search::hnsw_index::hnsw::{quantize, Hnsw};
use semantic_search::utils::{load_data, load_model};

#[allow(clippy::range_zip_with_len)]
fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    let do_quantize: bool = args[1] == "quantize";
    println!("do quantize (f32 to i8) : {:?}", do_quantize);

    let model: SentenceEmbeddingsModel = load_model();

    let data: Vec<String> = load_data();

    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(data.len());

    let bs: usize = 128;

    let pb;
    #[cfg(feature = "progress")]
    {
        pb = ProgressBar::new((data.len() / bs + 1) as u64);
    }
    #[cfg(not(feature = "progress"))]
    {
        pb = Instant::now();
    }

    for chunk in data.chunks(bs) {
        let embeds: Vec<Vec<f32>> = model.encode(chunk).unwrap();
        embeddings.extend(embeds);
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

    let nb_elem: usize = data.len();
    let max_nb_connection: u8 = 16;
    let ef_c: usize = 200;
    let nb_layer: u8 = 16;

    if !do_quantize {
        let index: Hnsw<f32, DistDot> =
            Hnsw::<f32, DistDot>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistDot {});

        let embeddings_indices: Vec<(&Vec<f32>, usize)> =
            embeddings.iter().zip(0..embeddings.len()).collect();

        let start: Instant = Instant::now();
        index.parallel_insert(&embeddings_indices);
        println!("parallel insert : {:.3?}", start.elapsed());

        _ = index.file_dump("news");
    } else {
        let index: Hnsw<i8, DistHamming> = Hnsw::<i8, DistHamming>::new(
            max_nb_connection,
            nb_elem,
            nb_layer,
            ef_c,
            DistHamming {},
        );

        let start: Instant = Instant::now();
        let quantized_embeddings: Vec<Vec<i8>> = embeddings.par_iter().map(quantize).collect();
        println!("quantize : {:.3?}", start.elapsed());

        let embeddings_indices: Vec<(&Vec<i8>, usize)> = quantized_embeddings
            .iter()
            .zip(0..embeddings.len())
            .collect();

        let start: Instant = Instant::now();
        index.parallel_insert(&embeddings_indices);
        println!("parallel insert : {:.3?}", start.elapsed());

        _ = index.file_dump("news_q");
    }

    Ok(())
}
