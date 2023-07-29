//! Api for external language.  
//! This file provides a trait to be used as an opaque pointer for C or Julia calls used in file
//! libext.rs

use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use std::io::BufWriter;
use std::path::PathBuf;

use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::hnsw_index::dist::Distance;
use crate::hnsw_index::hnsw::{Hnsw, Neighbour};
use crate::hnsw_index::hnswio::{DumpMode, HnswIO};

pub trait AnnT {
    /// type of data vectors
    type Val;

    ///
    #[allow(clippy::ptr_arg)]
    fn insert_data(&mut self, data: &Vec<Self::Val>, id: usize);

    ///
    #[allow(clippy::ptr_arg)]
    fn search_neighbours(&self, data: &Vec<Self::Val>, knbn: usize, ef_s: usize) -> Vec<Neighbour>;

    ///
    #[allow(clippy::ptr_arg)]
    fn parallel_insert_data(&mut self, data: &Vec<(&Vec<Self::Val>, usize)>);

    ///
    #[allow(clippy::ptr_arg)]
    fn parallel_search_neighbours(
        &self,
        data: &Vec<Vec<Self::Val>>,
        knbn: usize,
        ef_s: usize,
    ) -> Vec<Vec<Neighbour>>;

    /// dumps a data and graph in 2 files.
    /// Datas are dumped in file filename.hnsw.data and graph in filename.hnsw.graph
    #[allow(clippy::ptr_arg)]
    fn file_dump(&self, filename: &String) -> Result<i32, String>;
}

impl<T, D> AnnT for Hnsw<T, D>
where
    T: Serialize + DeserializeOwned + Clone + Send + Sync,
    D: Distance<T> + Send + Sync,
{
    type Val = T;

    ///
    fn insert_data(&mut self, data: &Vec<Self::Val>, id: usize) {
        self.insert((data, id));
    }

    ///
    fn search_neighbours(&self, data: &Vec<T>, knbn: usize, ef_s: usize) -> Vec<Neighbour> {
        self.search(data, knbn, ef_s)
    }

    fn parallel_insert_data(&mut self, data: &Vec<(&Vec<Self::Val>, usize)>) {
        self.parallel_insert(data);
    }

    fn parallel_search_neighbours(
        &self,
        data: &Vec<Vec<Self::Val>>,
        knbn: usize,
        ef_s: usize,
    ) -> Vec<Vec<Neighbour>> {
        self.parallel_search(data, knbn, ef_s)
    }

    /// The main entry point to do a dump.  
    /// It will generate two files one for the graph part of the data. The other for the real data
    /// points of the structure.
    fn file_dump(&self, filename: &String) -> Result<i32, String> {
        let graphpath: PathBuf = PathBuf::from(format!("{}.hnsw.graph", filename));
        let graph: File = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(graphpath)
            .unwrap();

        let datapath: PathBuf = PathBuf::from(format!("{}.hnsw.data", filename));
        let data: File = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(datapath)
            .unwrap();

        let mut graph_buf: BufWriter<File> = BufWriter::with_capacity(50_000_000, graph);
        let mut data_buf: BufWriter<File> = BufWriter::with_capacity(50_000_000, data);

        let res: Result<i32, String> = self.dump(DumpMode::Full, &mut graph_buf, &mut data_buf);

        graph_buf.flush().unwrap();
        data_buf.flush().unwrap();

        res
    }
} // end of impl block AnnT for Hnsw<T,D>
