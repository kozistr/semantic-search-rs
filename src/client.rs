use mimalloc::MiMalloc;
use std::error::Error;
use std::sync::mpsc;
use std::time::Duration;
use std::{env, process};
use tokio;

pub mod ss {
    tonic::include_proto!("ss");
}

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Debug, Clone)]
struct Config {
    n: usize,
    k: i32,
}
impl Config {
    fn new(args: &[String]) -> Result<Config, &'static str> {
        if args.len() < 2 {
            return Err("not enough arguments");
        }

        let n: usize = args[1].parse().unwrap();
        let k: i32 = args[2].parse().unwrap();

        Ok(Config { n, k })
    }
}

#[derive(Default, Debug)]
struct Metrics {
    model_lat: Vec<u64>,
    search_lat: Vec<u64>,
}

async fn execute(config: &Config) -> Result<Metrics, Box<dyn Error>> {
    let mut client: ss::inference_client::InferenceClient<tonic::transport::Channel> =
        ss::inference_client::InferenceClient::connect("http://127.0.0.1:50051").await?;

    let request: ss::PredictRequest = ss::PredictRequest {
        features: vec![
            ss::Features {
                query: "The story about the school life".to_string(),
                k: config.k,
            };
            1
        ],
    };

    // warm-up 10 times to load model & index files on the server-side
    for _ in 0..10 {
        _ = client.predict(request.clone()).await?;
    }

    let mut model_lat: Vec<u64> = vec![0u64; config.n];
    let mut search_lat: Vec<u64> = vec![0u64; config.n];

    for i in 1..config.n {
        let response: ss::PredictResponse = client.predict(request.clone()).await?.into_inner();

        model_lat[i] = response.model_latency;
        search_lat[i] = response.search_latency;
    }

    Ok(Metrics {
        model_lat,
        search_lat,
    })
}

#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("==================================================================");
    println!("Usage: client num_iters k");

    let args: Vec<String> = env::args().collect();

    let config: Config = Config::new(&args).unwrap_or_else(|err: &str| {
        println!("Problem parsing arguments: {}", err);
        process::exit(1);
    });

    println!("num_iters : {}, k : {}", config.n, config.k);

    println!("==================================================================");

    let (tx, rx) = mpsc::channel::<Metrics>();

    {
        let config: Config = config.clone();
        let tx: mpsc::Sender<Metrics> = tx.clone();

        tokio::spawn(async move {
            let metrics: Metrics = execute(&config).await.expect("ERROR!");
            tx.send(metrics).unwrap();
        });
    }

    let result = rx.recv().unwrap();

    let mut metrics: Metrics = Metrics::default();
    metrics.model_lat.extend(result.model_lat.iter());
    metrics.search_lat.extend(result.search_lat.iter());

    tokio::time::sleep(Duration::from_secs(1)).await;

    report(&config, &metrics);

    Ok(())
}

fn log_stats(i: usize, model_latencies: &Vec<u64>, search_latencies: &Vec<u64>, take: usize) {
    {
        let lats = model_latencies.iter().take(take);

        let mean: u64 = lats.clone().sum::<u64>() / i as u64;
        let max: u64 = *lats.clone().max().unwrap();
        let count: usize = lats.clone().collect::<Vec<&u64>>().len();

        let ps: Vec<String> = percentiles(vec![0.95, 0.99, 0.999], model_latencies, take)
            .iter()
            .map(|(p, x)| format!("p{:2.1}={:1.3}ms", 100.0 * p, *x as f64 * 1e-6))
            .collect();

        println!(
            "model latency  : {} Mean={:1.3}ms Max={:1.3}m Count={:>7} {}",
            i,
            mean as f64 * 1e-6,
            max as f64 * 1e-6,
            count,
            ps.join(" ")
        );
    }

    {
        let lats = search_latencies.iter().take(take);

        let mean: u64 = lats.clone().sum::<u64>() / i as u64;
        let max: u64 = *lats.clone().max().unwrap();
        let count: usize = lats.clone().collect::<Vec<&u64>>().len();

        let ps: Vec<String> = percentiles(vec![0.95, 0.99, 0.999], search_latencies, take)
            .iter()
            .map(|(p, x)| format!("p{:2.1}={:1.3}ms", 100.0 * p, *x as f64 * 1e-6))
            .collect();

        println!(
            "search latency : {} Mean={:1.3}ms Max={:1.3}m Count={:>7} {}",
            i,
            mean as f64 * 1e-6,
            max as f64 * 1e-6,
            count,
            ps.join(" ")
        );
    }
}

fn percentiles(ps: Vec<f64>, latencies: &Vec<u64>, take: usize) -> Vec<(f64, u64)> {
    let mut sorted: Vec<&u64> = latencies.iter().take(take).collect();
    sorted.sort();

    ps.iter()
        .map(|p: &f64| (*p, *sorted[(sorted.len() as f64 * p) as usize]))
        .collect()
}

fn report(config: &Config, metrics: &Metrics) {
    println!("REPORT =====================================================================");
    log_stats(config.n, &metrics.model_lat, &metrics.search_lat, config.n);
}
