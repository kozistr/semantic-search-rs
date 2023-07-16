//! defines a trait for filtering requests

use crate::hnsw_index::hnsw::DataId;

pub trait FilterT {
    fn hnsw_filter(&self, id: &DataId) -> bool;
}

impl FilterT for Vec<usize> {
    fn hnsw_filter(&self, id: &DataId) -> bool {
        return match &self.binary_search(id) {
            Ok(_) => true,
            _ => false,
        };
    }
}

impl<F> FilterT for F
where
    F: Fn(&DataId) -> bool,
{
    fn hnsw_filter(&self, id: &DataId) -> bool {
        return self(id);
    }
}
