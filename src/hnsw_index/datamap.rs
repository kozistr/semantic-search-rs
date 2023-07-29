//! This module provides a bidirectional link between a file in the format used for the dump of Data
//! vectors filling the Hnsw structure. We mmap the file and provide
//!     - a Hashmap from DataId to address
//!     - an interface for retrieving just data vectors loaded in the hnsw structure.
//!     - an interface for creating a Hnsw structure from the vectors stored in file
#![allow(unused)]
use std::default;
use std::fs::{File, Metadata, OpenOptions};
use std::io::{BufReader, Error};
use std::path::PathBuf;

use hashbrown::HashMap;
use mmap_rs::{Mmap, MmapOptions};

use crate::hnsw_index::hnsw::{DataId, Hnsw, Point, PointId};
use crate::hnsw_index::hnswio::{load_description, Description, MAGICDATAP};

/// This structure uses the data part of the dump of a Hnsw structure to retrieve the data.
/// The data is access via a mmap of the data file, so memory is spared at the expense of page
/// loading.
pub struct DataMap {
    /// File containing Points data
    datapath: PathBuf,
    /// The mmap structure
    mmap: Mmap,
    /// map a dataId to an address where we get a bson encoded vector of type T
    hmap: HashMap<DataId, usize>,
    /// type name of Data
    t_name: String,
    /// dimenstion
    dimension: usize,
} // end of DataMap

impl DataMap {
    pub fn new<T: Clone + Send + Sync>(dir: &str, filename: &str) -> Self {
        Self::from_hnswdump::<T>(dir, filename).unwrap()
    }

    // end of new

    // TODO: specifiy mmap option
    pub fn from_hnswdump<T: Clone + Send + Sync>(
        dir: &str,
        filename: &str,
    ) -> Result<DataMap, String> {
        let datapath: PathBuf = PathBuf::from(format!("{}{}.hnsw.data", dir, filename));

        let file: File = File::open(&datapath).unwrap();
        let filesize: usize = file.metadata().unwrap().len().try_into().unwrap();
        let offset: u64 = 0;

        let mmap_opt: MmapOptions<'_> = MmapOptions::new(filesize).unwrap();
        let mmap_opt: MmapOptions<'_> = unsafe { mmap_opt.with_file(&file, offset) };
        let mmap: Mmap = mmap_opt.map().unwrap_or_else(|_| {
            log::error!("could not memory map : {:?}", &datapath);
            std::process::exit(1);
        });

        // reload description to have data type
        let graphpath: PathBuf = PathBuf::from(format!("{}{}.hnsw.graph", dir, filename));
        let graphfile: File = OpenOptions::new().read(true).open(&graphpath).unwrap();
        let mut graph_in: BufReader<File> = BufReader::new(graphfile);

        // we need to call load_description first to get distance name
        let hnsw_description: Description = load_description(&mut graph_in).unwrap();
        if hnsw_description.format_version <= 2 {
            return Err(String::from(
                "from_hnsw::from_hnsw : data mapping is only possible for dumps with the version \
                 >= 0.1.20 of this crate",
            ));
        }

        let t_name: String = hnsw_description.get_typename();
        // get dimension as declared in description
        let descr_dimension: usize = hnsw_description.get_dimension();
        drop(graph_in);

        // check typename coherence
        if std::any::type_name::<T>() != t_name {
            return Err(String::from("type error"));
        }

        let mapped_slice: &[u8] = mmap.as_slice();

        // where are we in decoding mmap slice?
        let mut current_mmap_addr: usize = 0usize;

        // check magic
        let mut it_slice: [u8; 4] = [0u8; std::mem::size_of::<u32>()];
        it_slice.copy_from_slice(
            &mapped_slice[current_mmap_addr..current_mmap_addr + std::mem::size_of::<u32>()],
        );
        current_mmap_addr += std::mem::size_of::<u32>();
        let magic: u32 = u32::from_ne_bytes(it_slice);
        assert_eq!(magic, MAGICDATAP, "magic not equal to MAGICDATAP in mmap");

        // get dimension
        let mut it_slice: [u8; 8] = [0u8; std::mem::size_of::<usize>()];
        it_slice.copy_from_slice(
            &mapped_slice[current_mmap_addr..current_mmap_addr + std::mem::size_of::<usize>()],
        );
        current_mmap_addr += std::mem::size_of::<usize>();
        let dimension: usize = usize::from_ne_bytes(it_slice);
        if dimension != descr_dimension {
            return Err(String::from("description and data do not agree on dimension"));
        }

        // now we know that each record consists in
        //   - MAGICDATAP (u32), DataId  (u64), serialized_len (lenght in bytes * dimension)
        let record_size: usize = std::mem::size_of::<u32>()
            + 2 * std::mem::size_of::<u64>()
            + dimension * std::mem::size_of::<T>();
        let residual: usize = mmap.size() - current_mmap_addr;

        let nb_record: usize = residual / record_size;

        // allocate hmap with correct capacity
        let mut hmap: HashMap<DataId, usize> = HashMap::<DataId, usize>::with_capacity(nb_record);

        // fill hmap to have address of each data point in file
        let mut u32_slice: [u8; 4] = [0u8; std::mem::size_of::<u32>()];
        let mut u64_slice: [u8; 8] = [0u8; std::mem::size_of::<u64>()];

        // now we loop on records
        for i in 0..nb_record {
            // decode Magic
            u32_slice.copy_from_slice(
                &mapped_slice[current_mmap_addr..current_mmap_addr + std::mem::size_of::<u32>()],
            );
            current_mmap_addr += std::mem::size_of::<u32>();

            let magic: u32 = u32::from_ne_bytes(u32_slice);
            assert_eq!(magic, MAGICDATAP, "magic not equal to MAGICDATAP in mmap");

            // decode DataId
            u64_slice.copy_from_slice(
                &mapped_slice[current_mmap_addr..current_mmap_addr + std::mem::size_of::<u64>()],
            );
            current_mmap_addr += std::mem::size_of::<DataId>();
            let data_id: usize = DataId::from_ne_bytes(u64_slice);

            // Note we store address where we have to decode dimension*size_of::<T> and full bson
            // encoded vector
            hmap.insert(data_id, current_mmap_addr);

            // now read serialized length
            u64_slice.copy_from_slice(
                &mapped_slice[current_mmap_addr..current_mmap_addr + std::mem::size_of::<u64>()],
            );
            current_mmap_addr += std::mem::size_of::<u64>();
            let serialized_len: usize = u64::from_ne_bytes(u64_slice) as usize;

            current_mmap_addr += serialized_len;
        } // end of for on record

        Ok(DataMap { datapath, mmap, hmap, t_name, dimension: descr_dimension })
    }

    /// return the data corresponding to dataid. Access is done via mmap
    pub fn get_data<T: Clone + std::fmt::Debug>(&self, dataid: &DataId) -> Option<&[T]> {
        let address: usize = *self.hmap.get(dataid)?;

        let mut current_mmap_addr: usize = address;
        let mapped_slice: &[u8] = self.mmap.as_slice();

        let mut u64_slice: [u8; 8] = [0u8; std::mem::size_of::<u64>()];
        u64_slice.copy_from_slice(
            &mapped_slice[current_mmap_addr..current_mmap_addr + std::mem::size_of::<u64>()],
        );

        let serialized_len: usize = u64::from_ne_bytes(u64_slice) as usize;
        current_mmap_addr += std::mem::size_of::<u64>();

        let slice_t: &[T] = unsafe {
            std::slice::from_raw_parts(
                mapped_slice[current_mmap_addr..].as_ptr() as *const T,
                self.dimension,
            )
        };

        Some(slice_t)
    }
} // end of impl DataMap

//=====================================================================================

#[cfg(test)]
mod tests {

    use rand::distributions::{Distribution, Uniform};

    use super::*;
    use crate::hnsw_index::api::AnnT;
    use crate::hnsw_index::dist::DistL1;

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_file_mmap() {
        println!("\n\n test_file_mmap");
        log_init_test();
        // generate a random test
        let mut rng: rand::rngs::ThreadRng = rand::thread_rng();
        let unif: Uniform<f32> = Uniform::<f32>::new(0., 1.);
        // 1000 vectors of size 10 f32
        let nbcolumn: usize = 1000;
        let nbrow: usize = 10;
        let mut xsi: f32;
        let mut data: Vec<Vec<f32>> = Vec::with_capacity(nbcolumn);
        for j in 0..nbcolumn {
            data.push(Vec::with_capacity(nbrow));
            for _ in 0..nbrow {
                xsi = unif.sample(&mut rng);
                data[j].push(xsi);
            }
        }

        // define hnsw
        let ef_construct: usize = 25;
        let nb_connection: u8 = 10;
        let hnsw: Hnsw<f32, DistL1> =
            Hnsw::<f32, DistL1>::new(nb_connection, nbcolumn, 16, ef_construct, DistL1 {});
        for i in 0..data.len() {
            hnsw.insert((&data[i], i));
        }
        // some loggin info
        hnsw.dump_layer_info();

        // dump in a file. Must take care of name as tests runs in // !!!
        _ = hnsw.file_dump("mmap_test");

        let datamap: DataMap = DataMap::new::<i8>(".", "mmap_test");
    } // end of test_file_mmap
} // end of mod tests
