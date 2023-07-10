use std::error::Error;

use mimalloc::MiMalloc;

pub mod ss {
    tonic::include_proto!("ss");
}

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut client = ss::inference_client::InferenceClient::connect("127.0.0.1:50051").await?;

    let request = ss::PredictRequest {
        features: vec![
            ss::Features {
                query: "The story about the school life".to_string(),
                k: 10,
            };
            1
        ],
    };

    // warm up 10 times
    for _ in 0..10 {
        _ = client.predict(request.clone()).await?;
    }

    let result = client.predict(request.clone()).await?;
    println!("result : {:?}", result);

    Ok(())
}
