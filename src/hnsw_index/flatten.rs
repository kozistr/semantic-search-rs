//! This module provides conversion of a Point structure to a FlatPoint containing just the Id of a
//! point and those of its neighbours.
//! The whole Hnsw structure is then flattened into a Hashtable associating the data ID of a point
//! to its corresponding FlatPoint.   
//! It can be used, for example, when reloading only the graph part of the data to have knowledge
//! of relative proximity of points as described just by their DataId

use std::cmp::Ordering;

use hashbrown::HashMap;
use rayon::prelude::*;

use crate::hnsw_index::dist::Distance;
use crate::hnsw_index::hnsw::{DataId, Hnsw, Neighbour, Point, PointId};

// an ordering of Neighbour of a Point

impl PartialEq for Neighbour {
    fn eq(&self, other: &Neighbour) -> bool {
        self.distance == other.distance
    } // end eq
}

impl Eq for Neighbour {}

// order points by distance to self.
impl PartialOrd for Neighbour {
    fn partial_cmp(&self, other: &Neighbour) -> Option<Ordering> {
        Some(self.cmp(other))
    } // end cmp
} // end impl PartialOrd

impl Ord for Neighbour {
    fn cmp(&self, other: &Neighbour) -> Ordering {
        if !self.distance.is_nan() && !other.distance.is_nan() {
            self.distance.partial_cmp(&other.distance).unwrap()
        } else {
            panic!("got a NaN in a distance");
        }
    } // end cmp
}

/// a reduced version of point inserted in the Hnsw structure.
/// It contains original id of point as submitted to the struct Hnsw
/// an ordered (by distance) list of neighbours to the point
/// and it position in layers.
#[derive(Clone)]
pub struct FlatPoint {
    /// an id coming from client using hnsw, should identify point uniquely
    origin_id: DataId,
    /// a point id identifying point as stored in our structure
    p_id: PointId,
    /// neighbours info
    neighbours: Vec<Neighbour>,
}

impl FlatPoint {
    /// returns the neighbours orderded by distance.
    pub fn get_neighbours(&self) -> &Vec<Neighbour> {
        &self.neighbours
    }

    /// returns the origin id of the point
    pub fn get_id(&self) -> DataId {
        self.origin_id
    }

    ///
    pub fn get_p_id(&self) -> PointId {
        self.p_id
    }
} // end impl block for FlatPoint

fn flatten_point<T: Clone + Send + Sync>(point: &Point<T>) -> FlatPoint {
    let neighbours: Vec<Vec<Neighbour>> = point.get_neighborhood_id();

    // now we flatten neighbours
    let mut flat_neighbours: Vec<Neighbour> = neighbours
        .iter()
        .flat_map(|layer: &Vec<Neighbour>| layer.iter())
        .cloned()
        .collect();

    flat_neighbours.par_sort_unstable();

    FlatPoint {
        origin_id: point.get_origin_id(),
        p_id: point.get_point_id(),
        neighbours: flat_neighbours,
    }
} // end of flatten_point

/// A structure providing neighbourhood information of a point stored in the Hnsw structure given
/// its DataId. The structure uses the [FlatPoint] structure.  
/// This structure can be obtained by FlatNeighborhood::from<&Hnsw<T,D>>
pub struct FlatNeighborhood {
    hash_t: HashMap<DataId, FlatPoint>,
}

impl FlatNeighborhood {
    /// get neighbour of a point given its id.  
    /// The neighbours are sorted in increasing distance from data_id.
    pub fn get_neighbours(&self, p_id: DataId) -> Option<Vec<Neighbour>> {
        self.hash_t
            .get(&p_id)
            .map(|point: &FlatPoint| point.get_neighbours().clone())
    }
} // end impl block for FlatNeighborhood

impl<T: Clone + Send + Sync, D: Distance<T> + Send + Sync> From<&Hnsw<T, D>> for FlatNeighborhood {
    /// extract from the Hnsw strucure a hashtable mapping original DataId into a FlatPoint
    /// structure gathering its neighbourhood information. Useful after reloading from a dump
    /// with T=NoData and D = NoDist as points are then reloaded with neighbourhood information
    /// only.
    fn from(hnsw: &Hnsw<T, D>) -> Self {
        let mut hash_t: HashMap<usize, FlatPoint> = HashMap::new();

        let ptiter = hnsw.get_point_indexation().into_iter();

        for point in ptiter {
            let res_insert: Option<FlatPoint> =
                hash_t.insert(point.get_origin_id(), flatten_point(&point));

            if let Some(old_point) = res_insert {
                println!("2 points with same origin id {:?}", old_point.origin_id);
                log::error!("2 points with same origin id {:?}", old_point.origin_id);
            }
        }

        FlatNeighborhood { hash_t }
    }
} // e,d of Fom implementation

#[cfg(test)]

mod tests {

    use std::fs::{File, OpenOptions};
    use std::io::BufReader;
    use std::path::PathBuf;

    use rand::distributions::{Distribution, Uniform};

    use super::*;
    use crate::hnsw_index::api::AnnT;
    use crate::hnsw_index::dist::{DistL1, NoDist};
    use crate::hnsw_index::hnsw::{check_graph_equality, NoData};
    use crate::hnsw_index::hnswio::{load_description, load_hnsw, Description};

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_dump_reload_graph_flatten() {
        println!("\n\n test_dump_reload_graph_flatten");
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

        // get flat neighbours of point 3
        let neighborhood_before_dump: FlatNeighborhood = FlatNeighborhood::from(&hnsw);
        let nbg_2_before: Vec<Neighbour> = neighborhood_before_dump.get_neighbours(2).unwrap();
        println!("voisins du point 2 {:?}", nbg_2_before);

        // dump in a file. Must take care of name as tests runs in // !!!
        let fname: String = String::from("dumpreloadtestflat");
        let _res: Result<i32, String> = hnsw.file_dump(&fname);
        // This will dump in 2 files named dumpreloadtest.hnsw.graph and dumpreloadtest.hnsw.data
        //
        // reload
        log::debug!("\n\n  hnsw reload");

        // we will need a procedural macro to get from distance name to its instanciation.
        // from now on we test with DistL1
        let graphfname: String = String::from("dumpreloadtestflat.hnsw.graph");
        let graphpath: PathBuf = PathBuf::from(graphfname);
        let graphfileres: Result<File, std::io::Error> =
            OpenOptions::new().read(true).open(&graphpath);
        if graphfileres.is_err() {
            println!("test_dump_reload: could not open file {:?}", graphpath.as_os_str());
            std::panic::panic_any("test_dump_reload: could not open file".to_string());
        }
        let graphfile: File = graphfileres.unwrap();

        let datafname: String = String::from("dumpreloadtestflat.hnsw.data");
        let datapath: PathBuf = PathBuf::from(datafname);
        let datafileres: Result<File, std::io::Error> =
            OpenOptions::new().read(true).open(&datapath);
        if datafileres.is_err() {
            println!("test_dump_reload : could not open file {:?}", datapath.as_os_str());
            std::panic::panic_any("test_dump_reload : could not open file".to_string());
        }
        let datafile: File = datafileres.unwrap();

        let mut graph_in: BufReader<File> = BufReader::new(graphfile);
        let mut data_in: BufReader<File> = BufReader::new(datafile);

        // we need to call load_description first to get distance name
        let hnsw_description: Description = load_description(&mut graph_in).unwrap();
        let hnsw_loaded: Hnsw<NoData, NoDist> =
            load_hnsw(&mut graph_in, &hnsw_description, &mut data_in).unwrap();
        let neighborhood_after_dump: FlatNeighborhood = FlatNeighborhood::from(&hnsw_loaded);
        let nbg_2_after: Vec<Neighbour> = neighborhood_after_dump.get_neighbours(2).unwrap();
        println!("voisins du point 2 {:?}", nbg_2_after);

        // test equality of neighborhood
        assert_eq!(nbg_2_after.len(), nbg_2_before.len());
        for i in 0..nbg_2_before.len() {
            assert_eq!(nbg_2_before[i].p_id, nbg_2_after[i].p_id);
            assert_eq!(nbg_2_before[i].distance, nbg_2_after[i].distance);
        }
        check_graph_equality(&hnsw_loaded, &hnsw);
    } // end of test_dump_reload
} // end module test
