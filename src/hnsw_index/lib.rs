use env_logger::Builder;

#[macro_use]
extern crate lazy_static;

pub mod api;
pub mod datamap;
pub mod dist;
pub mod filter;
pub mod flatten;
pub mod hnsw;
pub mod hnswio;

lazy_static! {
    static ref LOG: u64 = {
        init_log()
    };
}

fn init_log() -> u64 {
    Builder::from_default_env().init();
    return 1;
}
