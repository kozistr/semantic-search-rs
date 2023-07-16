use anyhow::Result;
use std::net::SocketAddr;
use tokio;
use tonic::{transport::Server, Request, Response, Status};

use semantic_search::{
	search::search,
	ss::{
		inference_server::{Inference, InferenceServer},
		PredictRequest, PredictResponse,
	},
};

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
