pub mod hnsw_index;
pub mod search;
pub mod utils;

pub mod ss {
    use tonic;

    tonic::include_proto!("ss");
}
