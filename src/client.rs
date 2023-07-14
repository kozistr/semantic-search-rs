use anyhow::Result;
use mimalloc::MiMalloc;
use std::{
    env, process,
    sync::mpsc,
    time::{Duration, Instant},
};
use tokio;

pub mod ss {
    tonic::include_proto!("ss");
}

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Debug, Clone)]
struct Config {
    u: usize,
    n: usize,
    bs: usize,
    k: i32,
}
impl Config {
    fn new(args: &[String]) -> Result<Config, &'static str> {
        if args.len() < 4 {
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
    let mut client: ss::inference_client::InferenceClient<tonic::transport::Channel> =
        ss::inference_client::InferenceClient::connect("http://127.0.0.1:50051").await?;

    let requests: ss::PredictRequest = ss::PredictRequest {
        features: vec![
            ss::Features {
                query: "The story about the school life".to_string(),
            };
            config.bs
        ],
        k: config.k,
    };

    // warm-up 10 times to load model & index files on the server-side
    for _ in 0..10 {
        _ = client.predict(requests.clone()).await?;
    }

    let mut model_lat: Vec<u64> = vec![0u64; config.n];
    let mut search_lat: Vec<u64> = vec![0u64; config.n];
    let mut total_lat: Vec<u64> = vec![0u64; config.n];

    for i in 1..config.n {
        let start: Instant = Instant::now();
        let response: ss::PredictResponse = client.predict(requests.clone()).await?.into_inner();

        total_lat[i] = start.elapsed().as_nanos() as u64;
        model_lat[i] = response.model_latency;
        search_lat[i] = response.search_latency;
    }

    Ok(Metrics {
        model_lat,
        search_lat,
        total_lat,
    })
}

#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    let config: Config = Config::new(&args).unwrap_or_else(|err: &str| {
        println!("Problem parsing arguments: {}", err);
        println!("Usage: client num_users num_iters k");
        process::exit(1);
    });

    println!(
        "num_users : {}, num_iters : {}, k : {}",
        config.u, config.n, config.k
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

fn log_stats(description: &str, i: usize, latencies: &Vec<u64>, take: usize) {
    let lats = latencies.iter().take(take);

    let mean: u64 = lats.clone().sum::<u64>() / i as u64;
    let max: u64 = *lats.clone().max().unwrap();

    let ps: Vec<String> = percentiles(&[0.95, 0.99, 0.999], latencies, take)
        .iter()
        .map(|(p, x)| format!("p{:2.1}={:1.3}ms", 100.0 * p, *x as f64 * 1e-6))
        .collect();

    println!(
        "{} latency : {} mean={:1.3}ms max={:1.3}m {}",
        description,
        i,
        mean as f64 * 1e-6,
        max as f64 * 1e-6,
        ps.join(" ")
    );
}

fn percentiles(ps: &[f64], latencies: &Vec<u64>, take: usize) -> Vec<(f64, u64)> {
    let mut sorted: Vec<u64> = latencies.iter().take(take).copied().collect();
    sorted.sort();

    ps.iter()
        .map(|p: &f64| (*p, sorted[((sorted.len() as f64) * p) as usize]))
        .collect()
}

fn report(config: &Config, metrics: &Metrics) {
    println!("REPORT =====================================================================");
    log_stats("total", config.n, &metrics.total_lat, config.n);
    log_stats("model", config.n, &metrics.model_lat, config.n);
    log_stats("search", config.n, &metrics.search_lat, config.n);
}
