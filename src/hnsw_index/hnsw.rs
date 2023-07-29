//! A rust implementation of Approximate NN search from:  
//! Efficient and robust approximate nearest neighbour search using Hierarchical Navigable
//! small World graphs.
//! Yu. A. Malkov, D.A Yashunin 2016, 2018
use std::any::type_name;
use std::cmp::Ordering;
use std::collections::binary_heap::BinaryHeap;
#[allow(unused)]
use std::collections::HashSet;
use std::sync::{mpsc, Arc};

use dashmap::DashMap;
use hashbrown::HashMap;
use parking_lot::{Mutex, RwLock, RwLockReadGuard};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::hnsw_index::dist::Distance;
use crate::hnsw_index::filter::FilterT;

const MAX_QVALUE: f32 = 127.0f32;

// TODO
// Profiling.

/// This unit structure provides the type to instanciate Hnsw with,
/// to get reload of graph only in the the structure.
/// It must be associated to the unit structure dist::NoDist for the distance type to provide.
#[derive(Default, Clone, Copy, Serialize, Deserialize)]
pub struct NoData;

/// maximum number of layers
pub(crate) const NB_LAYER_MAX: u8 = 16; // so max layer is 15!!

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// The 2-uple represent layer as u8  and rank in layer as a i32 as stored in our structure
pub struct PointId(pub u8, pub i32);

/// this type is for an identificateur of each data vector, given by client.
/// Can be the rank of data in an array, a hash value or anything that permits
/// retrieving the data.
pub type DataId = usize;

pub type PointDistance<T> = Box<dyn Distance<T>>;

/// A structure containing internal pointId with distance to this pointId.
/// The order is given by ordering the distance to the point it refers to.
/// So points ordering has a meaning only has points refers to the same point
#[derive(Debug, Clone, Copy)]
pub struct PointIdWithOrder {
    /// the identificateur of the point for which we store a distance
    pub point_id: PointId,
    /// The distance to a reference point (not represented in the structure)
    pub dist_to_ref: f32,
}

impl PartialEq for PointIdWithOrder {
    fn eq(&self, other: &PointIdWithOrder) -> bool {
        self.dist_to_ref == other.dist_to_ref
    } // end eq
}

// order points by distance to self.
impl PartialOrd for PointIdWithOrder {
    fn partial_cmp(&self, other: &PointIdWithOrder) -> Option<Ordering> {
        self.dist_to_ref.partial_cmp(&other.dist_to_ref)
    } // end cmp
} // end impl PartialOrd

impl<T: Send + Sync + Clone + Copy> From<&PointWithOrder<T>> for PointIdWithOrder {
    fn from(point: &PointWithOrder<T>) -> PointIdWithOrder {
        PointIdWithOrder::new(point.point_ref.p_id, point.dist_to_ref)
    }
}

impl PointIdWithOrder {
    pub fn new(point_id: PointId, dist_to_ref: f32) -> Self {
        PointIdWithOrder { point_id, dist_to_ref }
    }
} // end of impl block

//=======================================================================================
/// The struct giving an answer point to a search request.
/// This structure is exported to other language API.
/// First field is origin id of the request point, second field is distance to request point
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct Neighbour {
    /// identification of data vector as given in initializing hnsw
    pub d_id: DataId,
    /// distance of neighbours
    pub distance: f32,
    /// point identification inside layers
    pub p_id: PointId,
}

impl Neighbour {
    pub fn new(d_id: DataId, distance: f32, p_id: PointId) -> Neighbour {
        Neighbour { d_id, distance, p_id }
    }

    /// retrieves original id of neighbour as given in hnsw initialization
    pub fn get_origin_id(&self) -> DataId {
        self.d_id
    }

    /// return the distance
    pub fn get_distance(&self) -> f32 {
        self.distance
    }
}

//=======================================================================================

type Neighbor<T> = Vec<Arc<PointWithOrder<T>>>;
type Neighbors<T> = Vec<Neighbor<T>>;

/// The basestructure representing a data point.  
/// Its constains data as coming from the client, its client id,  
/// and position in layer representation and neighbours.
// neighbours table : one vector by layer so neighbours is allocated to NB_LAYER_MAX
#[derive(Debug, Clone)]
pub struct Point<T: Clone + Send + Sync> {
    /// The data of this point, coming from hnsw client and associated to origin_id,
    v: Vec<T>,
    /// an id coming from client using hnsw, should identify point uniquely
    origin_id: DataId,
    /// a point id identifying point as stored in our structure
    p_id: PointId,
    /// neighbours info
    pub(crate) neighbours: Arc<RwLock<Neighbors<T>>>,
}

impl<T: Clone + Send + Sync> Point<T> {
    pub fn new(v: &[T], origin_id: usize, p_id: PointId) -> Self {
        let mut neighbours: Neighbors<T> = Vec::with_capacity(NB_LAYER_MAX as usize);
        for _ in 0..NB_LAYER_MAX {
            neighbours.push(Vec::<Arc<PointWithOrder<T>>>::new());
        }

        Point { v: v.to_vec(), origin_id, p_id, neighbours: Arc::new(RwLock::new(neighbours)) }
    }

    /// get a reference to vector data
    pub fn get_v(&self) -> &[T] {
        self.v.as_slice()
    }

    /// return coordinates in indexation
    pub fn get_point_id(&self) -> PointId {
        self.p_id
    }

    /// returns external (or client id) id of point
    pub fn get_origin_id(&self) -> usize {
        self.origin_id
    }

    /// returns for each layer, a vector Neighbour of a point, one vector by layer
    /// useful for extern crate only as it reallocates vectors
    pub fn get_neighborhood_id(&self) -> Vec<Vec<Neighbour>> {
        let ref_neighbours = self.neighbours.read();
        let nb_layer: usize = ref_neighbours.len();
        let mut neighborhood: Vec<Vec<Neighbour>> = Vec::<Vec<Neighbour>>::with_capacity(nb_layer);
        for i in 0..nb_layer {
            let mut neighbours: Vec<Neighbour> = Vec::<Neighbour>::new();
            let nb_ngbh: usize = ref_neighbours[i].len();
            if nb_ngbh > 0 {
                neighbours.reserve(nb_ngbh);
                for pointwo in &ref_neighbours[i] {
                    neighbours.push(Neighbour::new(
                        pointwo.point_ref.get_origin_id(),
                        pointwo.dist_to_ref,
                        pointwo.point_ref.get_point_id(),
                    ));
                }
            }
            neighborhood.push(neighbours);
        }
        neighborhood
    }

    /// prints minimal information on neighbours of point.
    pub fn debug_dump(&self) {
        println!(" \n dump of point id : {:?}", self.p_id);
        println!("\n origin id : {:?} ", self.origin_id);
        println!(" neighbours : ...");
        let ref_neighbours = self.neighbours.read();
        for i in 0..ref_neighbours.len() {
            if !ref_neighbours[i].is_empty() {
                println!("neighbours at layer {:?}", i);
                for n in &ref_neighbours[i] {
                    println!(" {:?}", n.point_ref.p_id);
                }
            }
        }
        println!(" neighbours dump : end");
    }
} // end of block

//===========================================================================================

/// A structure to store neighbours for of a point.
#[derive(Debug, Clone)]
pub(crate) struct PointWithOrder<T: Clone + Send + Sync> {
    /// the identificateur of the point for which we store a distance to a point for which
    ///  we made a request.
    point_ref: Arc<Point<T>>,
    /// The distance to a point_ref to the request point (not represented in the structure)
    dist_to_ref: f32,
}

impl<T: Clone + Send + Sync> PartialEq for PointWithOrder<T> {
    fn eq(&self, other: &PointWithOrder<T>) -> bool {
        self.dist_to_ref == other.dist_to_ref
    } // end eq
}

impl<T: Clone + Send + Sync> Eq for PointWithOrder<T> {}

// order points by distance to self.
impl<T: Clone + Send + Sync> PartialOrd for PointWithOrder<T> {
    fn partial_cmp(&self, other: &PointWithOrder<T>) -> Option<Ordering> {
        Some(self.cmp(other))
    } // end cmp
} // end impl PartialOrd

impl<T: Clone + Send + Sync> Ord for PointWithOrder<T> {
    fn cmp(&self, other: &PointWithOrder<T>) -> Ordering {
        if !self.dist_to_ref.is_nan() && !other.dist_to_ref.is_nan() {
            self.dist_to_ref.partial_cmp(&other.dist_to_ref).unwrap()
        } else {
            panic!("got a NaN in a distance");
        }
    } // end cmp
}

impl<T: Clone + Send + Sync> PointWithOrder<T> {
    pub fn new(point_ref: &Arc<Point<T>>, dist_to_ref: f32) -> Self {
        PointWithOrder { point_ref: Arc::clone(point_ref), dist_to_ref }
    }
} // end of impl block

//============================================================================================

//  LayerGenerator
use rand::distributions::Uniform;
use rand::prelude::*;

/// a struct to randomly generate a level for an item according to an exponential law
/// of parameter given by scale.
/// The distribution is constrained to be in [0..maxlevel[
pub struct LayerGenerator {
    rng: Arc<Mutex<rand::rngs::StdRng>>,
    unif: Uniform<f32>,
    scale: f32,
    maxlevel: usize,
}

impl LayerGenerator {
    pub fn new(max_nb_connection: usize, maxlevel: usize) -> Self {
        let scale: f32 = 1. / (max_nb_connection as f32).ln();
        LayerGenerator {
            rng: Arc::new(Mutex::new(StdRng::from_entropy())),
            unif: Uniform::<f32>::new(0., 1.),
            scale,
            maxlevel,
        }
    }

    // l=0 most densely packed layer
    // if S is scale we sample so that P(l=n) = exp(-n/S) - exp(- (n+1)/S)
    // with S = 1./ln(max_nb_connection) P(l >= maxlevel) = exp(-maxlevel * ln(max_nb_connection))
    // for nb_conn = 10, even with maxlevel = 10,  we get P(l >= maxlevel) = 1.e-13
    // In Malkov(2016) S = 1.0 / log(max_nb_connection)
    //
    /// generate a layer with given maxlevel. upper layers (higher index) are of decreasing
    /// probabilities. thread safe method.
    fn generate(&self) -> usize {
        let mut protected_rng = self.rng.lock();
        let xsi: f32 = protected_rng.sample(self.unif);
        let level: f32 = -xsi.ln() * self.scale;
        let mut ulevel: usize = level.floor() as usize;

        // we redispatch possibly sampled level  >= maxlevel to required range
        if ulevel >= self.maxlevel {
            // This occurs with very low probability. Cf commentary above.
            ulevel = protected_rng.sample(Uniform::<usize>::new(0, self.maxlevel));
        }
        ulevel
    }

    /// just to try some variations on exponential level sampling. Unused.
    pub fn set_scale_modification(&mut self, scale_modification: f32) {
        self.scale = 1. / ((1. / self.scale) + scale_modification.ln());
    }
} // end impl for LayerGenerator

// ====================================================================

/// A short-hand for points in a layer
type Layer<T> = Vec<Arc<Point<T>>>;

/// a structure for indexation of points in layer
#[allow(unused)]
pub struct PointIndexation<T: Clone + Send + Sync> {
    /// max number of connection for a point at a layer
    pub(crate) max_nb_connection: usize,
    ///
    pub(crate) max_layer: usize,
    /// needs at least one representation of points. points_by_layers\[i\] gives the points in
    /// layer i
    pub(crate) points_by_layer: Arc<RwLock<Vec<Layer<T>>>>,
    /// utility to generate a level
    pub(crate) layer_g: LayerGenerator,
    /// number of points in indexed structure
    pub(crate) nb_point: Arc<RwLock<usize>>,
    /// curent enter_point: an Arc RwLock on a possible Arc Point
    pub(crate) entry_point: Arc<RwLock<Option<Arc<Point<T>>>>>,
}

// A point indexation may contain circular references. To deallocate these after a point indexation
// goes out of scope, implement the Drop trait.

impl<T: Clone + Send + Sync> Drop for PointIndexation<T> {
    fn drop(&mut self) {
        // clear_neighborhood. There are no point in neighborhoods that are not referenced directly
        // in layers. so we cannot loose reference to a point by cleaning neighborhood
        fn clear_neighborhoods<T: Clone + Send + Sync>(init: &Point<T>) {
            let mut neighbours = init.neighbours.write();
            let nb_layer: usize = neighbours.len();
            for l in 0..nb_layer {
                neighbours[l].clear();
            }
            neighbours.clear();
        }

        // clear entry point
        if let Some(i) = self.entry_point.write().as_ref() {
            clear_neighborhoods(i.as_ref());
        }

        let nb_level: u8 = self.get_max_level_observed();
        for l in 0..=nb_level {
            let layer: &mut Vec<Arc<Point<T>>> = &mut self.points_by_layer.write()[l as usize];
            layer
                .into_par_iter()
                .for_each(|p: &mut Arc<Point<T>>| clear_neighborhoods(p));
            layer.clear();
        }

        drop(self.points_by_layer.write());
    } // end my drop
} // end implementation Drop

impl<T: Clone + Send + Sync> PointIndexation<T> {
    pub fn new(max_nb_connection: usize, max_layer: usize, max_elements: usize) -> Self {
        let mut points_by_layer: Vec<Vec<Arc<Point<T>>>> = Vec::with_capacity(max_layer);

        let max_layer_f32: f32 = max_layer as f32;
        for i in 0..max_layer {
            // recall that range are right extremeity excluded
            // compute fraction of points going into layer i and do expected memory reservation
            let frac: f32 =
                (-(i as f32) / max_layer_f32).exp() - (-((i + 1) as f32) / max_layer_f32);
            let expected_size: usize = (frac * max_elements as f32).round() as usize;
            points_by_layer.push(Vec::with_capacity(expected_size));
        }

        let layer_g: LayerGenerator = LayerGenerator::new(max_nb_connection, max_layer);

        PointIndexation {
            max_nb_connection,
            max_layer,
            points_by_layer: Arc::new(RwLock::new(points_by_layer)),
            layer_g,
            nb_point: Arc::new(RwLock::new(0)),
            entry_point: Arc::new(RwLock::new(None)),
        }
    }

    // end of new

    /// returns the maximum level of layer observed
    pub fn get_max_level_observed(&self) -> u8 {
        let opt = self.entry_point.read();
        match opt.as_ref() {
            Some(arc_point) => arc_point.p_id.0,
            None => 0,
        }
    }

    fn debug_dump(&self) {
        println!(" debug dump of PointIndexation");
        let max_level_observed: u8 = self.get_max_level_observed();
        // CAVEAT a lock once
        for l in 0..=max_level_observed as usize {
            println!(" layer {} : length : {} ", l, self.points_by_layer.read()[l].len());
        }
        println!(" debug dump of PointIndexation end");
    }

    /// real insertion of point in point indexation
    // generate a new Point/ArcPoint (with neigbourhood info empty) and store it in global table
    // The function is called by Hnsw insert method
    fn generate_new_point(&self, data: &[T], origin_id: usize) -> (Arc<Point<T>>, usize) {
        // get a write lock at the beginning of the function
        let level: usize = self.layer_g.generate();

        let new_point: Arc<Point<T>>;
        {
            // open a write lock on points_by_layer
            let mut points_by_layer_ref = self.points_by_layer.write();
            let mut p_id: PointId = PointId(level as u8, -1);
            p_id.1 = points_by_layer_ref[p_id.0 as usize].len() as i32;

            // make a Point and then an Arc<Point>
            let point: Point<T> = Point::new(data, origin_id, p_id);
            new_point = Arc::new(point);

            points_by_layer_ref[p_id.0 as usize].push(Arc::clone(&new_point));
        } // close write lock on points_by_layer

        let nb_point: usize;
        {
            let mut lock_nb_point = self.nb_point.write();
            *lock_nb_point += 1;
            nb_point = *lock_nb_point;
            if nb_point % 50000 == 0 {
                println!(" setting number of points {:?} ", nb_point);
            }
        }

        // Now possibly this is a point on a new layer that will have no neighbours in its layer
        (Arc::clone(&new_point), nb_point)
    }

    // end of insert

    /// check if entry_point is modified
    fn check_entry_point(&self, new_point: &Arc<Point<T>>) {
        // take directly a write lock so that we are sure nobody can change anything between read
        // and write of entry_point_id
        let mut entry_point_ref = self.entry_point.write();
        match entry_point_ref.as_ref() {
            Some(arc_point) => {
                if new_point.p_id.0 > arc_point.p_id.0 {
                    *entry_point_ref = Some(Arc::clone(new_point));
                }
            },
            None => {
                *entry_point_ref = Some(Arc::clone(new_point));
            },
        }
    }

    // end of check_entry_point

    /// returns the number of points in layered structure
    pub fn get_nb_point(&self) -> usize {
        *self.nb_point.read()
    }

    /// returns the number of points in a given layer, 0 on a bad layer num
    pub fn get_layer_nb_point(&self, layer: usize) -> usize {
        let nb_layer: usize = self.points_by_layer.read().len();
        if layer < nb_layer { self.points_by_layer.read()[layer].len() } else { 0 }
    }

    // end of get_layer_nb_point

    /// returns the size of data vector in graph if any, else return 0
    pub fn get_data_dimension(&self) -> usize {
        let ep = self.entry_point.read();
        match ep.as_ref() {
            Some(point) => point.get_v().len(),
            None => 0,
        }
    }

    /// returns (**by cloning**) the data inside a point given it PointId, or None if PointId is not
    /// coherent. Can be useful after reloading from a dump.   
    /// NOTE : This function should not be called during or before insertion in the structure is
    /// terminated as it uses read locks to access the inside of Hnsw structure.
    pub fn get_point_data(&self, p_id: &PointId) -> Option<Vec<T>> {
        if p_id.1 < 0 {
            return None;
        }

        let p: usize = std::convert::TryFrom::try_from(p_id.1).unwrap();
        let l: usize = p_id.0 as usize;

        if p_id.0 <= self.get_max_level_observed() && p < self.get_layer_nb_point(l) {
            Some(self.points_by_layer.read()[l][p].get_v().to_vec())
        } else {
            None
        }
    }

    // end of get_point_data

    /// returns (**by Arc::clone**) the point given it PointId, or None if PointId is not coherent.
    /// Can be useful after reloading from a dump.   
    /// NOTE : This function should not be called during or before insertion in the structure is
    /// terminated as it uses read locks to access the inside of Hnsw structure.
    #[allow(unused)]
    pub(crate) fn get_point(&self, p_id: &PointId) -> Option<Arc<Point<T>>> {
        if p_id.1 < 0 {
            return None;
        }

        let p: usize = std::convert::TryFrom::try_from(p_id.1).unwrap();
        let l: usize = p_id.0 as usize;

        if p_id.0 <= self.get_max_level_observed() && p < self.get_layer_nb_point(l) {
            Some(self.points_by_layer.read()[l][p].clone())
        } else {
            None
        }
    }

    // end of get_point

    /// get an iterator on the points stored in a given layer
    pub fn get_layer_iterator(&self, layer: usize) -> IterPointLayer<T> {
        IterPointLayer::new(self, layer)
    } // end of get_layer_iterator
} // end of impl PointIndexation

//============================================================================================

/// an iterator on points stored.
/// The iteration begins at level 0 (most populated level) and goes upward in levels.
/// The iterator takes a ReadGuard on the PointIndexation structure
pub struct IterPoint<'a, T: Clone + Send + Sync> {
    point_indexation: &'a PointIndexation<T>,
    pi_guard: RwLockReadGuard<'a, Vec<Layer<T>>>,
    layer: i64,
    slot_in_layer: i64,
}

impl<'a, T: Clone + Send + Sync> IterPoint<'a, T> {
    pub fn new(point_indexation: &'a PointIndexation<T>) -> Self {
        let pi_guard: RwLockReadGuard<Vec<Layer<T>>> = point_indexation.points_by_layer.read();
        IterPoint { point_indexation, pi_guard, layer: -1, slot_in_layer: -1 }
    }
} // end of block impl IterPoint

/// iterator for layer 0 to upper layer.
impl<'a, T: Clone + Send + Sync> Iterator for IterPoint<'a, T> {
    type Item = Arc<Point<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.layer == -1 {
            self.layer = 0;
            self.slot_in_layer = 0;
        }
        if (self.slot_in_layer as usize) < self.pi_guard[self.layer as usize].len() {
            let slot: usize = self.slot_in_layer as usize;
            self.slot_in_layer += 1;
            Some(self.pi_guard[self.layer as usize][slot].clone())
        } else {
            self.slot_in_layer = 0;
            self.layer += 1;

            // must reach a non empty layer if possible
            let entry_point_ref = self.point_indexation.entry_point.read();
            let points_by_layer = self.point_indexation.points_by_layer.read();
            let entry_point_level: u8 = entry_point_ref.as_ref().unwrap().p_id.0;
            while (self.layer as u8) <= entry_point_level
                && points_by_layer[self.layer as usize].is_empty()
            {
                self.layer += 1;
            }

            // now here either (self.layer as u8) > self.point_indexation.max_level_observed
            // or self.point_indexation.points_by_layer[self.layer as usize ].len() > 0
            if (self.layer as u8) <= entry_point_level {
                let slot: usize = self.slot_in_layer as usize;
                self.slot_in_layer += 1;
                Some(points_by_layer[self.layer as usize][slot].clone())
            } else {
                None
            }
        }
    } // end of next
} // end of impl Iterator

impl<'a, T: Clone + Send + Sync> IntoIterator for &'a PointIndexation<T> {
    type IntoIter = IterPoint<'a, T>;
    type Item = Arc<Point<T>>;

    fn into_iter(self) -> Self::IntoIter {
        IterPoint::new(self)
    }
} // end of IntoIterator for &'a PointIndexation<T>

/// An iterator on points stored in a given layer
/// The iterator stores a ReadGuard on the structure PointIndexation
pub struct IterPointLayer<'a, T: Clone + Send + Sync> {
    _point_indexation: &'a PointIndexation<T>,
    pi_guard: RwLockReadGuard<'a, Vec<Layer<T>>>,
    layer: usize,
    slot_in_layer: usize,
}

impl<'a, T: Clone + Send + Sync> IterPointLayer<'a, T> {
    pub fn new(point_indexation: &'a PointIndexation<T>, layer: usize) -> Self {
        let pi_guard: RwLockReadGuard<Vec<Layer<T>>> = point_indexation.points_by_layer.read();
        IterPointLayer { _point_indexation: point_indexation, pi_guard, layer, slot_in_layer: 0 }
    }
} // end of block impl IterPointLayer

/// iterator for layer 0 to upper layer.
impl<'a, T: Clone + Send + Sync> Iterator for IterPointLayer<'a, T> {
    type Item = Arc<Point<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.slot_in_layer < self.pi_guard[self.layer].len() {
            let slot: usize = self.slot_in_layer;
            self.slot_in_layer += 1;
            Some(self.pi_guard[self.layer][slot].clone())
        } else {
            None
        }
    } // end of next
} // end of impl Iterator

// ============================================================================================

// The fields are made pub(crate) to be able to initialize struct from hnswio
/// The Base structure for hnsw implementation.  
/// The main useful functions are : new, insert, insert_parallel, search, parallel_search and
/// file_dump as described in trait AnnT.  
///
/// Other functions are mainly for others crate to get access to some fields.
pub struct Hnsw<T: Clone + Send + Sync, D: Distance<T>> {
    /// asked number of candidates in search
    pub(crate) ef_construction: usize,
    /// maximum number of connection by layer for a point
    pub(crate) max_nb_connection: usize,
    /// flag to enforce that we have ef candidates as pruning strategy can discard some points
    /// Can be set to true with method :set_extend_candidates
    /// When set to true used only in base layer.
    pub(crate) extend_candidates: bool,
    /// defuault to false
    pub(crate) keep_pruned: bool,
    /// max layer , recall rust is in 0..maxlevel right bound excluded
    pub(crate) max_layer: usize,
    /// The global table containing points
    pub(crate) layer_indexed_points: PointIndexation<T>,
    /// dimension data stored in points
    #[allow(unused)]
    pub(crate) data_dimension: usize,
    /// distance between points. initialized at first insertion
    pub(crate) dist_f: D,
    /// insertion mode or searching mode. This flag prevents a internal thread to do a write when
    /// searching with other threads.
    pub(crate) searching: bool,
} // end of Hnsw

impl<T: Clone + Send + Sync, D: Distance<T> + Send + Sync> Hnsw<T, D> {
    /// allocation function  
    /// . max_nb_connection : number of neighbours stored, by layer, in tables. Must be less than
    ///   256.
    /// . ef_construction : controls numbers of neighbours explored during construction. See README
    ///   or paper.
    /// . max_elements : hint to speed up allocation tables. number of elements expected.
    /// . f : the distance function
    pub fn new(
        max_nb_connection: usize,
        max_elements: usize,
        max_layer: usize,
        ef_construction: usize,
        f: D,
    ) -> Self {
        let adjusted_max_layer: usize = (NB_LAYER_MAX as usize).min(max_layer);
        let layer_indexed_points: PointIndexation<T> =
            PointIndexation::<T>::new(max_nb_connection, adjusted_max_layer, max_elements);
        let extend_candidates: bool = false;
        let keep_pruned: bool = false;

        if max_nb_connection > 256 {
            println!("error max_nb_connection must be less equal than 256");
            std::process::exit(1);
        }

        log::info!("Hnsw max_nb_connection {:?}", max_nb_connection);
        log::info!("Hnsw nb elements {:?}", max_elements);
        log::info!("Hnsw ef_construction {:?}", ef_construction);
        log::info!("Hnsw distance {:?}", type_name::<D>());
        log::info!("Hnsw extend candidates {:?}", extend_candidates);

        Hnsw {
            max_nb_connection,
            ef_construction,
            extend_candidates,
            keep_pruned,
            max_layer: adjusted_max_layer,
            layer_indexed_points,
            data_dimension: 0,
            dist_f: f,
            searching: false,
        }
    }

    // end of new

    /// get ef_construction used in graph creation
    pub fn get_ef_construction(&self) -> usize {
        self.ef_construction
    }

    /// returns the maximum layer authorized in construction
    pub fn get_max_level(&self) -> usize {
        self.max_layer
    }

    /// return the maximum level reached in the layers.
    pub fn get_max_level_observed(&self) -> u8 {
        self.layer_indexed_points.get_max_level_observed()
    }

    /// returns the maximum of links between a point and others points in each layer
    pub fn get_max_nb_connection(&self) -> u8 {
        self.max_nb_connection as u8
    }

    /// returns number of points stored in hnsw structure
    pub fn get_nb_point(&self) -> usize {
        self.layer_indexed_points.get_nb_point()
    }

    /// set searching mode.  
    /// It is not possible to do parallel insertion and parallel searching simultaneously in
    /// different threads so to enable searching after parallel insertion the flag must be set
    /// to true. To resume parallel insertion reset the flag to false and so on.
    pub fn set_searching_mode(&mut self, flag: bool) {
        // must use an atomic!
        self.searching = flag;
    }

    /// get name if distance
    pub fn get_distance_name(&self) -> String {
        type_name::<D>().to_string()
    }

    /// set the flag asking to keep pruned vectors by Navarro's heuristic (see Paper).
    /// It can be useful for small datasets where the pruning can make it difficult
    /// to get the exact number of neighbours asked for.
    pub fn set_keeping_pruned(&mut self, flag: bool) {
        self.keep_pruned = flag;
    }

    /// retrieves the distance used in Hnsw construction
    pub fn get_distance(&self) -> &D {
        &self.dist_f
    }

    /// set extend_candidates to given flag. By default it is false.  
    /// Only used in the level 0 layer during insertion (see the paper)
    /// flag to enforce that we have ef candidates neighbours examined as pruning strategy
    /// can discard some points
    pub fn set_extend_candidates(&mut self, flag: bool) {
        self.extend_candidates = flag;
    }

    // multiplicative factor applied to default scale. Must between 0.5 and 1.
    // more  than 1. gives more occupied layers. This is just to experiment
    // parameters variations on the algorithm but not general use.
    #[allow(unused)]
    fn set_scale_modification(&mut self, scale_modification: f32) {
        println!(
            "\n scale modification factor {:?}, scale value : {:?} (factor must be between 0.5 \
             and 2.)",
            scale_modification, self.layer_indexed_points.layer_g.scale
        );

        if (0.5..=2.).contains(&scale_modification) {
            self.layer_indexed_points
                .layer_g
                .set_scale_modification(scale_modification);
            println!(" new scale value {:?}", self.layer_indexed_points.layer_g.scale);
        } else {
            println!(
                "\n scale modificationarg {:?} not valid , factor must be between 0.5 and 2.)",
                scale_modification
            );
        }
    }

    // end of set_scale_modification

    // here we could pass a point_id_with_order instead of entry_point_id: PointId
    // The efficacity depends on greedy part depends on how near entry point is from point.
    // ef is the number of points to return
    // The method returns a BinaryHeap with positive distances. The caller must transforms it
    // according its need
    //** NOTE: the entry point is pushed into returned point at the beginning of the function, but
    //** in fact entry_point is in a layer with higher (one more) index than the argument layer.
    //** If the greedy search matches a sufficiently large number of points nearer to point
    //** searched (arg point) than entry_point it will finally get popped up from the heap of
    //** returned points but otherwise it will stay in the binary heap and so we can have a
    //** point in neighbours that is in fact in a layer above the one we search in.
    //** The guarantee is that the binary heap will return points in layer
    //** with a larger index, although we can expect that most often (at least in densely populated
    //** layers) the returned points will be found in searched layer
    /// Greedy algorithm n° 2 in Malkov paper.
    /// search in a layer (layer) for the ef points nearest a point to be inserted in hnsw.
    fn search_layer(
        &self,
        point: &[T],
        entry_point: Arc<Point<T>>,
        ef: usize,
        layer: u8,
        filter: Option<&dyn FilterT>,
    ) -> BinaryHeap<Arc<PointWithOrder<T>>> {
        // here we allocate a binary_heap on values not on reference beccause we want to return
        // log2(skiplist_size) must be greater than 1.
        let skiplist_size: usize = ef.max(2);

        // we will store positive distances in this one
        let mut return_points: BinaryHeap<Arc<PointWithOrder<T>>> =
            BinaryHeap::<Arc<PointWithOrder<T>>>::with_capacity(skiplist_size);

        if self.layer_indexed_points.points_by_layer.read()[layer as usize].is_empty() {
            // at the beginning we can have nothing in layer
            return return_points;
        }
        if entry_point.p_id.1 < 0 {
            return return_points;
        }

        // initialize visited points
        let dist_to_entry_point: f32 = self.dist_f.eval(point, &entry_point.v);

        // keep a list of id visited
        let mut visited_point_id: HashMap<PointId, Arc<Point<T>>> =
            HashMap::<PointId, Arc<Point<T>>>::new();
        visited_point_id.insert(entry_point.p_id, Arc::clone(&entry_point));

        let mut candidate_points: BinaryHeap<Arc<PointWithOrder<T>>> =
            BinaryHeap::<Arc<PointWithOrder<T>>>::with_capacity(skiplist_size);
        candidate_points.push(Arc::new(PointWithOrder::new(&entry_point, -dist_to_entry_point)));
        return_points.push(Arc::new(PointWithOrder::new(&entry_point, dist_to_entry_point)));

        // at the beginning candidate_points contains point passed as arg in layer entry_point_id.0
        while !candidate_points.is_empty() {
            // get nearest point in candidate_points
            let c: Arc<PointWithOrder<T>> = candidate_points.pop().unwrap();
            assert!(c.dist_to_ref <= 0.);

            // f farthest point to
            let f: &Arc<PointWithOrder<T>> = return_points.peek().unwrap();
            assert!(f.dist_to_ref >= 0.);

            if -c.dist_to_ref > f.dist_to_ref {
                // this comparison requires that we are sure that distances compared are distances
                // to the same point : This is the case we compare distance to point
                // passed as arg.
                if filter.is_none() || (filter.is_some() && return_points.len() >= ef) {
                    return return_points;
                }
            }

            // now we scan neighborhood of c in layer and increment visited_point, candidate_points
            // and optimize candidate_points so that it contains points with lowest distances to
            // point arg
            let neighbours_c_l: &Neighbor<T> = &c.point_ref.neighbours.read()[layer as usize];

            for e in neighbours_c_l {
                // HERE WE SEE THAT neighbours should be stored as PointIdWithOrder !!
                // CAVEAT what if several point_id with same distance to ref point?
                if !visited_point_id.contains_key(&e.point_ref.p_id) {
                    visited_point_id.insert(e.point_ref.p_id, Arc::clone(&e.point_ref));

                    let f_opt: Option<&Arc<PointWithOrder<T>>> = return_points.peek();
                    if f_opt.is_none() {
                        return return_points;
                    }

                    let f: &Arc<PointWithOrder<T>> = f_opt.unwrap();
                    let e_dist_to_p: f32 = self.dist_f.eval(point, &e.point_ref.v);
                    let f_dist_to_p: f32 = f.dist_to_ref;
                    if e_dist_to_p < f_dist_to_p || return_points.len() < ef {
                        let e_prime: Arc<PointWithOrder<T>> =
                            Arc::new(PointWithOrder::new(&e.point_ref, e_dist_to_p));

                        candidate_points
                            .push(Arc::new(PointWithOrder::new(&e.point_ref, -e_dist_to_p)));
                        if filter.is_none() {
                            return_points.push(Arc::clone(&e_prime));
                        } else {
                            let id: usize = e_prime.point_ref.get_origin_id();
                            if filter.as_ref().unwrap().hnsw_filter(&id) {
                                if return_points.len() == 1 {
                                    let only_id: usize =
                                        return_points.peek().unwrap().point_ref.origin_id;
                                    if !filter.as_ref().unwrap().hnsw_filter(&only_id) {
                                        return_points.clear()
                                    }
                                }
                                return_points.push(Arc::clone(&e_prime))
                            }
                        }
                        if return_points.len() > ef {
                            return_points.pop();
                        }
                    } // end if e.dist_to_ref < f.dist_to_ref
                }
            } // end of for on neighbours_c
        } // end of while in candidates

        return_points
    }

    // end of search_layer

    /// insert a tuple (&Vec, usize) with its external id as given by the client.
    ///  The insertion method gives the point an internal id.
    #[inline]
    pub fn insert(&self, datav_with_id: (&Vec<T>, usize)) {
        self.insert_slice(((datav_with_id.0.as_slice()), datav_with_id.1))
    }

    // Hnsw insert.
    ///  Insert a data slice with its external id as given by the client.   
    ///  The insertion method gives the point an internal id.  
    ///  The slice insertion makes integration with ndarray crate easier than the vector insertion
    pub fn insert_slice(&self, data_with_id: (&[T], usize)) {
        let (data, origin_id) = data_with_id;
        let keep_pruned: bool = self.keep_pruned;

        // insert in indexation and get point_id adn generate a new entry_point if necessary
        let (new_point, point_rank) = self
            .layer_indexed_points
            .generate_new_point(data, origin_id);

        // now real work begins
        // allocate a binary heap
        let level: u8 = new_point.p_id.0;
        let mut enter_point_copy: Option<Arc<Point<T>>> = None;
        let mut max_level_observed: u8 = 0;

        // entry point has been set in
        {
            // I open a read lock on an option
            if let Some(arc_point) = self.layer_indexed_points.entry_point.read().as_ref() {
                enter_point_copy = Some(Arc::clone(arc_point));
                if point_rank == 1 {
                    return;
                }
                max_level_observed = enter_point_copy.as_ref().unwrap().p_id.0;
            }
        }

        if enter_point_copy.is_none() {
            self.layer_indexed_points.check_entry_point(&new_point);
            return;
        }

        let mut dist_to_entry: f32 = self
            .dist_f
            .eval(data, &enter_point_copy.as_ref().unwrap().v);

        // we go from self.max_level_observed to level+1 included
        for l in ((level + 1)..(max_level_observed + 1)).rev() {
            // CAVEAT could bypass when layer empty, avoid  allocation..
            let mut sorted_points: BinaryHeap<Arc<PointWithOrder<T>>> =
                self.search_layer(data, Arc::clone(enter_point_copy.as_ref().unwrap()), 1, l, None);

            if sorted_points.len() > 1 {
                panic!(
                    "in insert : search_layer layer {:?}, returned {:?} points ",
                    l,
                    sorted_points.len()
                );
            }

            // the heap conversion is useless beccause of the preceding test.
            // sorted_points = from_positive_binaryheap_to_negative_binary_heap(&sorted_points);
            if let Some(ep) = sorted_points.pop() {
                // useful for projecting lower layer to upper layer. keep track of points
                // encountered.
                if new_point.neighbours.read()[l as usize].len()
                    < self.get_max_nb_connection() as usize
                {
                    new_point.neighbours.write()[l as usize].push(Arc::clone(&ep));
                }

                // get the lowest distance point
                let tmp_dist: f32 = self.dist_f.eval(data, &ep.point_ref.v);
                if tmp_dist < dist_to_entry {
                    enter_point_copy = Some(Arc::clone(&ep.point_ref));
                    dist_to_entry = tmp_dist;
                }
            }
        }

        // now enter_point_id_copy contains id of nearest
        // now loop down to 0
        for l in (0..level + 1).rev() {
            let ef: usize = self.ef_construction;
            // when l == level, we cannot get new_point in sorted_points as it is seen only from
            // declared neighbours
            let mut sorted_points: BinaryHeap<Arc<PointWithOrder<T>>> = self.search_layer(
                data,
                Arc::clone(enter_point_copy.as_ref().unwrap()),
                ef,
                l,
                None,
            );

            sorted_points = from_positive_binaryheap_to_negative_binary_heap(&sorted_points);
            if !sorted_points.is_empty() {
                let nb_conn: usize;
                let extend_c: bool;
                if l == 0 {
                    nb_conn = 2 * self.max_nb_connection;
                    extend_c = self.extend_candidates;
                } else {
                    nb_conn = self.max_nb_connection;
                    extend_c = false;
                }
                let mut neighbours: Vec<Arc<PointWithOrder<T>>> =
                    Vec::<Arc<PointWithOrder<T>>>::with_capacity(nb_conn);

                self.select_neighbours(
                    data,
                    &mut sorted_points,
                    nb_conn,
                    extend_c,
                    l,
                    keep_pruned,
                    &mut neighbours,
                );

                // sort neighbours
                neighbours.par_sort_unstable();

                // we must add bidirectional from data i.e new_point_id to neighbours
                new_point.neighbours.write()[l as usize] = neighbours.clone();

                // this reverse neighbour update could be done here but we put it at end to gather
                // all code requiring a mutex guard for multi threading.
                // update ep for loop iteration. As we sorted neighbours the nearest
                if !neighbours.is_empty() {
                    enter_point_copy = Some(Arc::clone(&neighbours[0].point_ref));
                }
            }
        } // for l

        // new_point has been inserted at the beginning in table
        // so that we can call reverse_update_neighborhoodwe consitently
        // now reverse update of neighbours.
        self.reverse_update_neighborhood_simple(Arc::clone(&new_point));

        self.layer_indexed_points.check_entry_point(&new_point);
    }

    // end of insert

    /// Insert in parallel a slice of Vec\<T\> each associated to its id.    
    /// It uses Rayon for threading so the number of insertions asked for must be large enough to be
    /// efficient. Typically 1000 * the number of threads.  
    /// Many consecutive parallel_insert can be done, so the size of vector inserted in one
    /// insertion can be optimized.
    pub fn parallel_insert(&self, datas: &Vec<(&Vec<T>, usize)>) {
        datas.par_iter().for_each(|&item| self.insert(item));
    }

    // end of parallel_insert

    /// Insert in parallel slices of \[T\] each associated to its id.    
    /// It uses Rayon for threading so the number of insertions asked for must be large enough to be
    /// efficient. Typically 1000 * the number of threads.  
    /// Facilitates the use with the ndarray crate as we can extract slices (for data in contiguous
    /// order) from Array.
    pub fn parallel_insert_slice(&self, datas: &Vec<(&[T], usize)>) {
        datas.par_iter().for_each(|&item| self.insert_slice(item));
    }

    // end of parallel_insert

    /// insert new_point in neighbourhood info of point
    fn reverse_update_neighborhood_simple(&self, new_point: Arc<Point<T>>) {
        let level: u8 = new_point.p_id.0;
        for l in (0..level + 1).rev() {
            for q in &new_point.neighbours.read()[l as usize] {
                if new_point.p_id != q.point_ref.p_id {
                    // as new point is in global table, do not loop and deadlock!!
                    let q_point: &Arc<Point<T>> = &q.point_ref;
                    let mut q_point_neighbours = q_point.neighbours.write();
                    let n_to_add: PointWithOrder<T> =
                        PointWithOrder::<T>::new(&Arc::clone(&new_point), q.dist_to_ref);

                    // must be sure that we add a point at the correct level. See the comment to
                    // search_layer! this ensures that reverse updating do not
                    // add problems.
                    let l_n: usize = n_to_add.point_ref.p_id.0 as usize;
                    let already: Option<usize> =
                        q_point_neighbours[l_n]
                            .iter()
                            .position(|old: &Arc<PointWithOrder<T>>| {
                                old.point_ref.p_id == new_point.p_id
                            });
                    if already.is_some() {
                        continue;
                    }

                    q_point_neighbours[l_n].push(Arc::new(n_to_add));
                    let nbn_at_l: usize = q_point_neighbours[l_n].len();
                    // if l < level, update upward chaining, insert does a sort! t_q has a neighbour
                    // not yet in global table of points!

                    // TODO optimize threshold
                    let threshold_shrinking: usize =
                        if l_n > 0 { self.max_nb_connection } else { 2 * self.max_nb_connection };

                    let shrink: bool = nbn_at_l > threshold_shrinking;
                    {
                        // sort and shring if necessary
                        q_point_neighbours[l_n].par_sort_unstable();
                        if shrink {
                            q_point_neighbours[l_n].pop();
                        }
                    }
                } // end protection against point identity
            }
        }
    }

    // end of reverse_update_neighborhood_simple

    pub fn get_point_indexation(&self) -> &PointIndexation<T> {
        &self.layer_indexed_points
    }

    // This is best explained in : Navarro. Searching in metric spaces by spatial approximation.
    /// simplest searh neighbours
    // The binary heaps here is with negative distance sorted.
    #[allow(clippy::too_many_arguments)]
    fn select_neighbours(
        &self,
        data: &[T],
        candidates: &mut BinaryHeap<Arc<PointWithOrder<T>>>,
        nb_neighbours_asked: usize,
        extend_candidates_asked: bool,
        layer: u8,
        keep_pruned: bool,
        neighbours_vec: &mut Vec<Arc<PointWithOrder<T>>>,
    ) {
        neighbours_vec.clear();

        // we will extend if we do not have enough candidates and it is explicitly asked in arg
        let mut extend_candidates: bool = false;
        if candidates.len() <= nb_neighbours_asked {
            if !extend_candidates_asked {
                // just transfer taking care of signs
                while !candidates.is_empty() {
                    let p: Arc<PointWithOrder<T>> = candidates.pop().unwrap();
                    assert!(-p.dist_to_ref >= 0.);

                    neighbours_vec
                        .push(Arc::new(PointWithOrder::new(&p.point_ref, -p.dist_to_ref)));
                }
                return;
            } else {
                extend_candidates = true;
            }
        }

        if extend_candidates {
            let mut candidates_set: HashMap<PointId, Arc<Point<T>>> =
                HashMap::<PointId, Arc<Point<T>>>::new();
            for c in candidates.iter() {
                candidates_set.insert(c.point_ref.p_id, Arc::clone(&c.point_ref));
            }

            let mut new_candidates_set: HashMap<PointId, Arc<Point<T>>> =
                HashMap::<PointId, Arc<Point<T>>>::new();

            // get a list of all neighbours of candidates
            for (_p_id, p_point) in candidates_set.iter() {
                let n_p_layer: &Vec<Arc<PointWithOrder<T>>> =
                    &p_point.neighbours.read()[layer as usize];
                for q in n_p_layer {
                    if !candidates_set.contains_key(&q.point_ref.p_id)
                        && !new_candidates_set.contains_key(&q.point_ref.p_id)
                    {
                        new_candidates_set.insert(q.point_ref.p_id, Arc::clone(&q.point_ref));
                    }
                }
            } // end of for p

            for (_p_id, p_point) in new_candidates_set.iter() {
                let dist_topoint: f32 = self.dist_f.eval(data, &p_point.v);
                candidates.push(Arc::new(PointWithOrder::new(p_point, -dist_topoint)));
            }
        } // end if extend_candidates

        let mut discarded_points: BinaryHeap<Arc<PointWithOrder<T>>> =
            BinaryHeap::<Arc<PointWithOrder<T>>>::new();
        while !candidates.is_empty() && neighbours_vec.len() < nb_neighbours_asked {
            // compare distances of e to data. we do not need to recompute dists!
            if let Some(e_p) = candidates.pop() {
                assert!(e_p.dist_to_ref <= 0.);

                let mut e_to_insert: bool = true;
                let e_point_v: &Vec<T> = &e_p.point_ref.v;

                // is e_p the nearest to reference? data than to previous neighbours
                if !neighbours_vec.is_empty() {
                    e_to_insert = !neighbours_vec.iter().any(|d: &Arc<PointWithOrder<T>>| {
                        self.dist_f.eval(e_point_v, &(d.point_ref.v)) <= -e_p.dist_to_ref
                    });
                }
                if e_to_insert {
                    neighbours_vec
                        .push(Arc::new(PointWithOrder::new(&e_p.point_ref, -e_p.dist_to_ref)));
                } else {
                    // ep is taken from a binary heap, so it has a negative sign, we keep its sign
                    // to store it in another binary heap will possibly need to retain the best ones
                    // from the discarde binaryHeap
                    if keep_pruned {
                        discarded_points
                            .push(Arc::new(PointWithOrder::new(&e_p.point_ref, e_p.dist_to_ref)));
                    }
                }
            }
        }
        // now this part of neighbours is the most interesting and is distance sorted.

        // not pruned are at the end of neighbours_vec which is not re-sorted , but discarded are
        // sorted.
        if keep_pruned {
            while !discarded_points.is_empty() && neighbours_vec.len() < nb_neighbours_asked {
                let best_point: Arc<PointWithOrder<T>> = discarded_points.pop().unwrap();
                assert!(best_point.dist_to_ref <= 0.);

                // do not forget to reverse sign
                neighbours_vec.push(Arc::new(PointWithOrder::new(
                    &best_point.point_ref,
                    -best_point.dist_to_ref,
                )));
            }
        };
    }

    // end of select_neighbours

    /// A utility to get printed info on how many points there are in each layer.
    pub fn dump_layer_info(&self) {
        self.layer_indexed_points.debug_dump();
    }

    // search the first knbn nearest neigbours of a data, but can modify ef for layer > 1
    // This function return Vec<Arc<PointWithOrder<T>>>
    // The parameter ef controls the width of the search in the lowest level, it must be greater
    // than number of neighbours asked. A rule of thumb could be between knbn and max_nb_connection.
    #[allow(unused)]
    fn search_general(&self, data: &[T], knbn: usize, ef_arg: usize) -> Vec<Neighbour> {
        let mut entry_point: Arc<Point<T>>;
        {
            // a lock on an option an a Arc<Point>
            let entry_point_opt_ref = self.layer_indexed_points.entry_point.read();
            if entry_point_opt_ref.is_none() {
                return Vec::<Neighbour>::new();
            } else {
                entry_point = Arc::clone((*entry_point_opt_ref).as_ref().unwrap());
            }
        }

        let mut dist_to_entry: f32 = self.dist_f.eval(data, &entry_point.as_ref().v);
        for layer in (1..=entry_point.p_id.0).rev() {
            let mut neighbours: BinaryHeap<Arc<PointWithOrder<T>>> =
                self.search_layer(data, Arc::clone(&entry_point), 1, layer, None);

            neighbours = from_positive_binaryheap_to_negative_binary_heap(&neighbours);
            if let Some(entry_point_tmp) = neighbours.pop() {
                // get the lowest distance point.
                let tmp_dist: f32 = self.dist_f.eval(data, &entry_point_tmp.point_ref.v);
                if tmp_dist < dist_to_entry {
                    entry_point = Arc::clone(&entry_point_tmp.point_ref);
                    dist_to_entry = tmp_dist;
                }
            }
        }

        // ef must be greater than knbn. Possibly it should be between knbn and
        // self.max_nb_connection
        let ef: usize = ef_arg.max(knbn);
        // now search with asked ef in layer 0
        let neighbours_heap: BinaryHeap<Arc<PointWithOrder<T>>> =
            self.search_layer(data, entry_point, ef, 0, None);

        // go from heap of points with negative dist to a sorted vec of increasing points with > 0
        // distances.
        let neighbours: Vec<Arc<PointWithOrder<T>>> = neighbours_heap.into_sorted_vec();

        // get the min of K and ef points into a vector.
        let last: usize = knbn.min(ef).min(neighbours.len());

        neighbours[0..last]
            .iter()
            .map(|p: &Arc<PointWithOrder<T>>| {
                Neighbour::new(
                    p.as_ref().point_ref.origin_id,
                    p.as_ref().dist_to_ref,
                    p.as_ref().point_ref.p_id,
                )
            })
            .collect()
    }

    // end of knn_search

    /// a filtered version of [`Self::search`].  
    /// A filter can be added to the search to get nodes with a particular property or id
    /// constraint. See examples in filter.rs
    pub fn search_filter(
        &self,
        data: &[T],
        knbn: usize,
        ef_arg: usize,
        filter: Option<&dyn FilterT>,
    ) -> Vec<Neighbour> {
        let entry_point: Arc<Point<T>>;
        {
            // a lock on an option an a Arc<Point>
            let entry_point_opt_ref = self.layer_indexed_points.entry_point.read();
            if entry_point_opt_ref.is_none() {
                return Vec::<Neighbour>::new();
            } else {
                entry_point = Arc::clone((*entry_point_opt_ref).as_ref().unwrap());
            }
        }

        let mut dist_to_entry: f32 = self.dist_f.eval(data, &entry_point.as_ref().v);
        let mut pivot: Arc<Point<T>> = Arc::clone(&entry_point);
        let mut new_pivot: Option<Arc<Point<T>>> = None;

        for layer in (1..=entry_point.p_id.0).rev() {
            let mut has_changed: bool = false;
            // search in stored neighbours
            {
                let neighbours: &Vec<Arc<PointWithOrder<T>>> =
                    &pivot.neighbours.read()[layer as usize];
                for n in neighbours {
                    // get the lowest  distance point.
                    let tmp_dist: f32 = self.dist_f.eval(data, &n.point_ref.v);
                    if tmp_dist < dist_to_entry {
                        new_pivot = Some(Arc::clone(&n.point_ref));
                        has_changed = true;
                        dist_to_entry = tmp_dist;
                    }
                } // end of for on neighbours
            }
            if has_changed {
                pivot = Arc::clone(new_pivot.as_ref().unwrap());
            }
        } // end on for on layers

        // ef must be greater than knbn. Possibly it should be between knbn and
        // self.max_nb_connection
        let ef: usize = ef_arg.max(knbn);
        // now search with asked ef in layer 0
        let neighbours_heap: BinaryHeap<Arc<PointWithOrder<T>>> =
            self.search_layer(data, pivot, ef, 0, filter);

        // go from heap of points with negative dist to a sorted vec of increasing points with > 0
        // distances.
        let neighbours: Vec<Arc<PointWithOrder<T>>> = neighbours_heap.into_sorted_vec();

        // get the min of K and ef points into a vector.
        let last: usize = knbn.min(ef).min(neighbours.len());

        neighbours[0..last]
            .iter()
            .map(|p: &Arc<PointWithOrder<T>>| {
                Neighbour::new(
                    p.as_ref().point_ref.origin_id,
                    p.as_ref().dist_to_ref,
                    p.as_ref().point_ref.p_id,
                )
            })
            .collect()
    }

    // end of search_filter

    #[inline]
    pub fn search_possible_filter(
        &self,
        data: &[T],
        knbn: usize,
        ef_arg: usize,
        filter: Option<&dyn FilterT>,
    ) -> Vec<Neighbour> {
        self.search_filter(data, knbn, ef_arg, filter)
    }

    /// search the first knbn nearest neigbours of a data and returns a Vector of Neighbour.   
    /// The parameter ef controls the width of the search in the lowest level, it must be greater
    /// than number of neighbours asked.  
    /// A rule of thumb could be between knbn and max_nb_connection.
    pub fn search(&self, data: &[T], knbn: usize, ef_arg: usize) -> Vec<Neighbour> {
        self.search_possible_filter(data, knbn, ef_arg, None)
    }

    #[allow(dead_code)]
    fn search_with_id(
        &self,
        request: (usize, &Vec<T>),
        knbn: usize,
        ef: usize,
    ) -> (usize, Vec<Neighbour>) {
        (request.0, self.search(request.1, knbn, ef))
    }

    /// knbn is the number of nearest neigbours asked for. Returns for each data vector
    /// a Vector of Neighbour
    pub fn parallel_search(
        &self,
        datas: &Vec<Vec<T>>,
        knbn: usize,
        ef: usize,
    ) -> Vec<Vec<Neighbour>> {
        let (sender, receiver) = mpsc::channel();

        // make up requests
        let nb_request: usize = datas.len();
        let requests: Vec<(usize, &Vec<T>)> = (0..nb_request).zip(datas.iter()).collect();

        requests.par_iter().for_each_with(
            sender,
            |s: &mut mpsc::Sender<(usize, Vec<Neighbour>)>, item: &(usize, &Vec<T>)| {
                s.send(self.search_with_id(*item, knbn, ef)).unwrap()
            },
        );

        let req_res: Vec<(usize, Vec<Neighbour>)> = receiver.iter().collect();

        // now sort to respect the key order of input
        let mut answers: Vec<Vec<Neighbour>> = Vec::<Vec<Neighbour>>::with_capacity(datas.len());

        // get a map from request id to rank
        let req_hash: DashMap<usize, usize> = DashMap::<usize, usize>::with_capacity(req_res.len());

        (0..req_res.len()).into_par_iter().for_each(|i: usize| {
            req_hash.insert(req_res[i].0, i);
        });

        (0..datas.len()).for_each(|i: usize| {
            let answer_i: usize = *req_hash.get(&i).unwrap();
            answers.push((req_res[answer_i].1).clone());
        });

        answers
    }

    // end of insert_parallel
} // end of Hnsw

/// quantize from f32 into i8 vector
#[allow(unused)]
pub fn quantize(vector: &Vec<f32>) -> Vec<i8> {
    // assume the given vector is l2 normalized vector.
    let mut v: Vec<i8> = Vec::with_capacity(vector.len());
    v.extend(vector.iter().copied().map(|x: f32| (x * MAX_QVALUE) as i8));
    v
}

// end of quantize

/// quantize from f32 into i8 vector
#[allow(unused)]
pub fn quantize_slice(vector: &[f32]) -> Vec<i8> {
    // assume the given vector is l2 normalized vector.
    let mut v: Vec<i8> = Vec::with_capacity(vector.len());
    v.extend(vector.iter().copied().map(|x: f32| (x * MAX_QVALUE) as i8));
    v
}

// end of quantize_slice

// This function takes a binary heap with points declared with a negative distance
// and returns a vector of points with their correct positive distance to some reference distance
// The vector is sorted by construction
#[allow(unused)]
fn from_negative_binaryheap_to_sorted_vector<T: Send + Sync + Copy>(
    heap_points: &BinaryHeap<Arc<PointWithOrder<T>>>,
) -> Vec<Arc<PointWithOrder<T>>> {
    heap_points
        .iter()
        .map(|p: &Arc<PointWithOrder<T>>| {
            assert!(p.dist_to_ref <= 0.);
            Arc::new(PointWithOrder::new(&p.point_ref, -p.dist_to_ref))
        })
        .collect()
}

// This function takes a binary heap with points declared with a positive distance
// and returns a binary_heap of points with their correct negative distance to some reference
// distance
fn from_positive_binaryheap_to_negative_binary_heap<T: Send + Sync + Clone>(
    positive_heap: &BinaryHeap<Arc<PointWithOrder<T>>>,
) -> BinaryHeap<Arc<PointWithOrder<T>>> {
    positive_heap
        .iter()
        .map(|p: &Arc<PointWithOrder<T>>| {
            assert!(p.dist_to_ref >= 0.);
            Arc::new(PointWithOrder::new(&p.point_ref, -p.dist_to_ref))
        })
        .collect()
}

// essentialy to check dump/reload conssistency
// in fact checks only equality of graph
#[allow(unused)]
pub(crate) fn check_graph_equality<T1, D1, T2, D2>(hnsw1: &Hnsw<T1, D1>, hnsw2: &Hnsw<T2, D2>)
where
    T1: Copy + Clone + Send + Sync,
    D1: Distance<T1> + Default + Send + Sync,
    T2: Copy + Clone + Send + Sync,
    D2: Distance<T2> + Default + Send + Sync,
{
    assert_eq!(hnsw1.get_nb_point(), hnsw2.get_nb_point());

    // check for entry point
    assert!(
        hnsw1.layer_indexed_points.entry_point.read().is_some()
            || hnsw1.layer_indexed_points.entry_point.read().is_some(),
        "one entry point is None"
    );

    let ep1_read = hnsw1.layer_indexed_points.entry_point.read();
    let ep2_read = hnsw2.layer_indexed_points.entry_point.read();
    let ep1: &Arc<Point<T1>> = ep1_read.as_ref().unwrap();
    let ep2: &Arc<Point<T2>> = ep2_read.as_ref().unwrap();
    assert_eq!(
        ep1.origin_id, ep2.origin_id,
        "different entry points {:?} {:?}",
        ep1.origin_id, ep2.origin_id
    );
    assert_eq!(ep1.p_id, ep2.p_id, "origin id {:?} ", ep1.origin_id);

    // check layers
    let layers_1 = hnsw1.layer_indexed_points.points_by_layer.read();
    let layers_2 = hnsw2.layer_indexed_points.points_by_layer.read();

    let mut nb_point_checked: usize = 0;
    let mut nb_neighbours_checked: i32 = 0;
    for i in 0..NB_LAYER_MAX as usize {
        assert_eq!(layers_1[i].len(), layers_2[i].len());
        for j in 0..layers_1[i].len() {
            let p1: &Arc<Point<T1>> = &layers_1[i][j];
            let p2: &Arc<Point<T2>> = &layers_2[i][j];
            assert_eq!(p1.origin_id, p2.origin_id);
            assert_eq!(p1.p_id, p2.p_id, "\n checking origin_id point {:?} ", p1.origin_id);

            nb_point_checked += 1;

            // check neighborhood
            let nbgh1 = p1.neighbours.read();
            let nbgh2 = p2.neighbours.read();
            assert_eq!(nbgh1.len(), nbgh2.len());

            for k in 0..nbgh1.len() {
                assert_eq!(nbgh1[k].len(), nbgh2[k].len());
                for l in 0..nbgh1[k].len() {
                    assert_eq!(nbgh1[k][l].point_ref.origin_id, nbgh2[k][l].point_ref.origin_id);
                    assert_eq!(nbgh1[k][l].point_ref.p_id, nbgh2[k][l].point_ref.p_id);

                    // CAVEAT for precision with f32
                    assert_eq!(nbgh1[k][l].dist_to_ref, nbgh2[k][l].dist_to_ref);
                    nb_neighbours_checked += 1;
                }
            }
        } // end of for j
    } // end of for i
    assert_eq!(nb_point_checked, hnsw1.get_nb_point());
} // end of check_reload

#[cfg(test)]
mod tests {

    use rand::distributions::Uniform;

    use super::*;
    use crate::hnsw_index::dist;

    #[test]

    fn test_iter_point() {
        //
        println!("\n\n test_iter_point");
        //
        let mut rng: ThreadRng = rand::thread_rng();
        let unif: Uniform<f32> = Uniform::<f32>::new(0., 1.);
        let nbcolumn: usize = 5000;
        let nbrow: usize = 10;
        let mut xsi: f32;
        let mut data: Vec<Vec<f32>> = Vec::with_capacity(nbcolumn);
        for j in 0..nbcolumn {
            data.push(Vec::with_capacity(nbrow));
            for _ in 0..nbrow {
                xsi = rng.sample(unif);
                data[j].push(xsi);
            }
        }
        // check insertion
        let ef_construct: usize = 25;
        let nb_connection: usize = 10;
        let hns: Hnsw<f32, dist::DistL1> = Hnsw::<f32, dist::DistL1>::new(
            nb_connection,
            nbcolumn,
            16,
            ef_construct,
            dist::DistL1 {},
        );
        for i in 0..data.len() {
            hns.insert((&data[i], i));
        }

        hns.dump_layer_info();

        // now check iteration
        let mut ptiter = hns.get_point_indexation().into_iter();
        let mut nb_dumped: usize = 0;
        loop {
            if let Some(_point) = ptiter.next() {
                nb_dumped += 1;
            } else {
                break;
            }
        } // end while
        assert_eq!(nb_dumped, nbcolumn);
    } // end of test_iter_point

    #[test]
    fn test_iter_layerpoint() {
        //
        println!("\n\n test_iter_point");
        //
        let mut rng: ThreadRng = rand::thread_rng();
        let unif: Uniform<f32> = Uniform::<f32>::new(0., 1.);
        let nbcolumn: usize = 5000;
        let nbrow: usize = 10;
        let mut xsi: f32;
        let mut data: Vec<Vec<f32>> = Vec::with_capacity(nbcolumn);
        for j in 0..nbcolumn {
            data.push(Vec::with_capacity(nbrow));
            for _ in 0..nbrow {
                xsi = rng.sample(unif);
                data[j].push(xsi);
            }
        }
        // check insertion
        let ef_construct: usize = 25;
        let nb_connection: usize = 10;
        let hns: Hnsw<f32, dist::DistL1> = Hnsw::<f32, dist::DistL1>::new(
            nb_connection,
            nbcolumn,
            16,
            ef_construct,
            dist::DistL1 {},
        );
        for i in 0..data.len() {
            hns.insert((&data[i], i));
        }

        hns.dump_layer_info();
        // now check iteration
        let layer_num: usize = 0;
        let nbpl: usize = hns.get_point_indexation().get_layer_nb_point(layer_num);
        let mut layer_iter = hns.get_point_indexation().get_layer_iterator(layer_num);
        //
        let mut nb_dumped: usize = 0;
        loop {
            if let Some(_point) = layer_iter.next() {
                //    println!("point : {:?}", _point.p_id);
                nb_dumped += 1;
            } else {
                break;
            }
        } // end while
        println!("test_iter_layerpoint : nb point in layer {} , nb found {}", nbpl, nb_dumped);
        //
        assert_eq!(nb_dumped, nbpl);
    } // end of test_iter_layerpoint
} // end of module test
