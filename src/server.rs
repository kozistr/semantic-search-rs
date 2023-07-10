use tonic::{transport::Server, Request, Response, Status};

mod search;
use search::search;

pub mod cb {
    tonic::include_proto!("search");
}

#[derive(Debug, Default)]
pub struct VectorSearchService {}

#[tonic::async_trait]
impl search::inference_server::Inference for VectorSearchService {
    async fn predict(
        &self,
        request: Request<search::PredictRequest>,
    ) -> Result<Response<search::PredictResponse>, Status> {
        let reply = search(request.into_inner());

        Ok(Response::new(reply))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "127.0.0.1:50051".parse()?;
    let service: VectorSearchService = VectorSearchService::default();

    let server = Server::builder()
        .add_service(search::inference_server::VectorSearchService::new(service))
        .serve(addr)
        .await?;

    println!("SERVER : {:?}", server);

    Ok(())
}
