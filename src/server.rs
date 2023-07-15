use anyhow::Result;
use std::net::SocketAddr;
use tokio;
use tonic::{transport::Server, Request, Response, Status};

use semantic_search::search::search;

pub mod ss {
    tonic::include_proto!("ss");
}

#[derive(Debug, Default)]
pub struct VectorSearchService {}

#[tonic::async_trait]
impl ss::inference_server::Inference for VectorSearchService {
    async fn predict(
        &self,
        request: Request<ss::PredictRequest>,
    ) -> Result<Response<ss::PredictResponse>, Status> {
        let reply: ss::PredictResponse = search(request.into_inner());

        Ok(Response::new(reply))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let addr: SocketAddr = "127.0.0.1:50051".parse()?;
    let service: VectorSearchService = VectorSearchService::default();

    let server = Server::builder()
        .add_service(ss::inference_server::InferenceServer::new(service))
        .serve(addr)
        .await?;

    println!("SERVER : {:?}", server);

    Ok(())
}
