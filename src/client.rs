use std::error::Error;

pub mod ss {
    tonic::include_proto!("ss");
}

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    let mut client = ss::inference_client::VectorSearchService::connect("127.0.0.1:50051").await?;

    let request = ss::PredictRequest {
        features: vec![
            ss::Features {
                query: "The story about the school life",
                k: 10,
            };
            8
        ],
    };

    // warm up 10 times
    for _ in 0..10 {
        _ = client.search(request.clone()).await?;
    }

    let result = client.search(request.clone()).await?;
    println!("result : {:?}", result);

    Ok(())
}
