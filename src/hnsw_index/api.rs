//! Api for external language.  
//! This file provides a trait to be used as an opaque pointer for C or Julia calls used in file
//! libext.rs

use std::fs::OpenOptions;
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
    fn insert_data(&mut self, data: &Vec<Self::Val>, id: usize);
    ///
    fn search_neighbours(&self, data: &Vec<Self::Val>, knbn: usize, ef_s: usize) -> Vec<Neighbour>;
    ///
    fn parallel_insert_data(&mut self, data: &Vec<(&Vec<Self::Val>, usize)>);
    ///
    fn parallel_search_neighbours(
        &self,
        data: &Vec<Vec<Self::Val>>,
        knbn: usize,
        ef_s: usize,
    ) -> Vec<Vec<Neighbour>>;
    /// dumps a data and graph in 2 files.
    /// Datas are dumped in file filename.hnsw.data and graph in filename.hnsw.graph
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
        log::debug!("\n in file_dump : {:?}", filename);

        let mut graphname: String = filename.clone();
        graphname.push_str(".hnsw.graph");

        let graphpath: PathBuf = PathBuf::from(graphname);
        let fileres: Result<std::fs::File, std::io::Error> = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&graphpath);

        if fileres.is_err() {
            log::error!("api::file_dump could not open file {:?}", graphpath.as_os_str());
            println!("api::file_dump: could not open file {:?}", graphpath.as_os_str());
            return Err("api::file_dump could not open file".to_string());
        }

        let graphfile: std::fs::File = fileres.unwrap();

        let mut dataname: String = filename.clone();
        dataname.push_str(".hnsw.data");

        let datapath: PathBuf = PathBuf::from(dataname);
        let fileres: Result<std::fs::File, std::io::Error> = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&datapath);

        if fileres.is_err() {
            println!("api::file_dumpcould not open file {:?}", datapath.as_os_str());
            return Err("api::file_dump could not open file".to_string());
        }
        let datafile: std::fs::File = fileres.unwrap();
        let mut graphbufw: BufWriter<std::fs::File> =
            BufWriter::with_capacity(50_000_000, graphfile);
        let mut databufw: BufWriter<std::fs::File> = BufWriter::with_capacity(50_000_000, datafile);
        let res: Result<i32, String> = self.dump(DumpMode::Full, &mut graphbufw, &mut databufw);

        graphbufw.flush().unwrap();
        databufw.flush().unwrap();
        log::debug!("\n end of dump");

        res
    } // end of dump
} // end of impl block AnnT for Hnsw<T,D>

// macro export makes the macro export t the root of the crate
#[macro_export]
macro_rules! mapdist_t(
    ("DistL1")       => (crate::dist::DistL1);
    ("DistL2")       => (crate::dist::DistL2);
    ("DistL2")       => (crate::dist::DistL2);
    ("DistDot")      => (crate::dist::DistDot);
    ("DistHamming")  => (crate::dist::DistHamming);
    ("DistJaccard")  => (crate::dist::DistJaccard);
    ("DistPtr")      => (crate::dist::DistPtr);
    ("DistLevenshtein") => (crate::dist::DistLevenshtein);
    ("DistJensenShannon") => (crate::dist::DistJensenShannon);
    ("DistHellinger") => (crate::dist::DistHellinger);
);
