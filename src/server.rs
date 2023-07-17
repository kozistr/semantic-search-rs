use std::net::SocketAddr;

use anyhow::Result;
use semantic_search::search::search;
use semantic_search::ss::inference_server::{Inference, InferenceServer};
use semantic_search::ss::{PredictRequest, PredictResponse};
use tokio;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

#[derive(Debug, Default)]
pub struct VectorSearchService {}

#[tonic::async_trait]
impl Inference for VectorSearchService {
    async fn predict(
        &self,
        request: Request<PredictRequest>,
    ) -> Result<Response<PredictResponse>, Status> {
        let reply: PredictResponse = search(request.into_inner());

        Ok(Response::new(reply))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let addr: SocketAddr = "127.0.0.1:50051".parse()?;
    let service: VectorSearchService = VectorSearchService::default();

    let server = Server::builder()
        .add_service(InferenceServer::new(service))
        .serve(addr)
        .await?;

    println!("SERVER : {:?}", server);

    Ok(())
}
