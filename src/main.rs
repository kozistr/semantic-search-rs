use std::time::Instant;
use std::{env, process};

use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel;
use semantic_search::hnsw_index::dist::DistDot;
use semantic_search::hnsw_index::hnsw::{quantize, Hnsw, Neighbour};
use semantic_search::utils::{load_data, load_index, load_model, load_quantize_index};

fn percentiles(ps: &[f32], lats: &Vec<u64>) -> Vec<(f32, u64)> {
    ps.iter()
        .map(|p: &f32| (*p, lats[((lats.len() as f32) * p) as usize]))
        .collect()
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        println!("Usage: main query [full or quantize]");
        process::exit(1);
    }

    let query: String = args[1].clone();
    let do_quantize: bool = args[2] == "quantize";

    println!("query : {:?}", query);
    println!("do quantize : {:?}", do_quantize);

    let data: Vec<String> = load_data();
    let model: SentenceEmbeddingsModel = load_model();

    let query_embedding: Vec<Vec<f32>> = model.encode(&[query]).unwrap();

    let neighbors: Vec<Neighbour> = if !do_quantize {
        let index: Hnsw<f32, DistDot> = load_index("news");

        index.search(&query_embedding[0], 10, 30)
    } else {
        let index: Hnsw<i8, DistDot> = load_quantize_index("news");

        let query_embedding: Vec<i8> = quantize(&query_embedding[0]);

        index.search(query_embedding.as_slice(), 10, 30)
    };

    for (k, neighbor) in neighbors.iter().enumerate() {
        println!("top {} | id : {}, dist : {}", k + 1, neighbor.d_id, neighbor.distance);
        println!("{}", data[neighbor.d_id]);
    }

    {
        // let index: Hnsw<f32, DistDot> = load_index("news");
        let index: Hnsw<i8, DistDot> = load_quantize_index("news");
        let query_embedding: Vec<i8> = quantize(&query_embedding[0]);

        let n: usize = 2000;

        for bs in [1024, 2048, 4096, 8192] {
            let query_embeddings: Vec<Vec<i8>> = vec![query_embedding.clone(); bs];
            // let query_embeddings: Vec<Vec<f32>> = vec![query_embedding[0].clone(); bs];
            let mut search_lat: Vec<u64> = vec![0u64; n];

            (0..n).for_each(|i: usize| {
                let start: Instant = Instant::now();
                _ = index.parallel_search(&query_embeddings, 10, 30);
                search_lat[i] = start.elapsed().as_nanos() as u64;
            });

            let mut lats: Vec<u64> = search_lat.clone();
            lats.sort_unstable();

            let mean: f64 = (lats.clone().iter().sum::<u64>() / n as u64) as f64 * 1e-6;
            let max: f64 = *lats.clone().last().unwrap() as f64 * 1e-6;

            let ps: Vec<String> = percentiles(&[0.5, 0.95, 0.99, 0.999], &lats)
                .iter()
                .map(|(p, x)| format!("p{:2.1}={:1.3} ms", 100.0 * p, *x as f64 * 1e-6))
                .collect();

            println!(
                "bs : {} mean={:1.3} ms {} max={:1.3} ms QPS={:?}",
                bs,
                mean,
                ps.join(" "),
                max,
                (1000. * (bs as f64 / mean)) as i32,
            );
        }
    }
}
