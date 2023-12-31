use std::sync::mpsc;
use std::time::{Duration, Instant};
use std::{env, process};

use anyhow::Result;
use semantic_search::ss::inference_client::InferenceClient;
use semantic_search::ss::{Features, PredictRequest, PredictResponse};
use semantic_search::utils::log_stats;

#[derive(Debug, Clone)]
struct Config {
    u: usize,
    n: usize,
    bs: usize,
    k: i32,
}
impl Config {
    fn new(args: &[String]) -> Result<Config, &'static str> {
        if args.len() < 5 {
            return Err("not enough arguments");
        }

        let u: usize = args[1].parse().unwrap();
        let n: usize = args[2].parse().unwrap();
        let bs: usize = args[3].parse().unwrap();
        let k: i32 = args[4].parse().unwrap();

        Ok(Config { u, n, bs, k })
    }
}

#[derive(Default, Debug)]
struct Metrics {
    model_lat: Vec<u64>,
    search_lat: Vec<u64>,
    total_lat: Vec<u64>,
}

async fn execute(config: &Config) -> Result<Metrics> {
    let mut client: InferenceClient<tonic::transport::Channel> =
        InferenceClient::connect("http://127.0.0.1:50051").await?;

    let requests: PredictRequest = PredictRequest {
        features: vec![Features { query: "The story about the school life".to_owned() }; config.bs],
        k: config.k,
    };

    // warm-up 11 times to load model & index files on the server-side
    for _ in 0..11 {
        _ = client.predict(requests.clone()).await?;
    }

    let mut model_lat: Vec<u64> = vec![0u64; config.n];
    let mut search_lat: Vec<u64> = vec![0u64; config.n];
    let mut total_lat: Vec<u64> = vec![0u64; config.n];

    for i in 1..config.n {
        let start: Instant = Instant::now();
        let response: PredictResponse = client.predict(requests.clone()).await?.into_inner();

        total_lat[i] = start.elapsed().as_nanos() as u64;
        model_lat[i] = response.model_latency;
        search_lat[i] = response.search_latency;
    }

    Ok(Metrics { model_lat, search_lat, total_lat })
}

#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    let config: Config = Config::new(&args).unwrap_or_else(|err: &str| {
        println!("Problem parsing arguments: {}", err);
        println!("Usage: client num_users num_iters bs k");
        process::exit(1);
    });

    println!(
        "num_users : {}, num_iters : {}, bs : {}, k : {}",
        config.u, config.n, config.bs, config.k
    );

    let (tx, rx) = mpsc::channel::<Metrics>();

    for _ in 0..config.u {
        let config: Config = config.clone();
        let tx: mpsc::Sender<Metrics> = tx.clone();

        tokio::spawn(async move {
            let metrics: Metrics = execute(&config).await.expect("ERROR!");
            tx.send(metrics).unwrap();
        });
    }

    let mut metrics: Metrics = Metrics::default();
    for _ in 0..config.u {
        let result: Metrics = rx.recv().unwrap();
        metrics.model_lat.extend(result.model_lat.iter());
        metrics.search_lat.extend(result.search_lat.iter());
        metrics.total_lat.extend(result.total_lat.iter());
    }

    tokio::time::sleep(Duration::from_secs(1)).await;

    report(&config, &metrics);

    Ok(())
}

fn report(config: &Config, metrics: &Metrics) {
    println!("REPORT =====================================================================");
    log_stats("total", config.n, config.bs, &metrics.total_lat);
    log_stats("model", config.n, config.bs, &metrics.model_lat);
    log_stats("search", config.n, config.bs, &metrics.search_lat);
}
