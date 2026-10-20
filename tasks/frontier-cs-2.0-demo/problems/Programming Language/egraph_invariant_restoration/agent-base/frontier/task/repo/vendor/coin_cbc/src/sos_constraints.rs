use std::os::raw::c_int;
use std::convert::TryInto;
use crate::raw::SOSConstraintType;
use crate::Col;

/// Represents a group of multiple special ordered set constraints
#[derive(Clone, Debug)]
pub(crate) struct SOSConstraints {
    // It is important that these fields remain private in order to ensure
    // memory safety of the API. They can only be mutated together through the public
    // methods of this struct.
    row_starts: Vec<c_int>,
    col_indices: Vec<c_int>,
    weights: Vec<f64>,
}

impl SOSConstraints {
    /// Add a constraint to the group of constraints
    pub fn add_constraint_with_weights<I: Iterator<Item=(Col, f64)>>(
        &mut self,
        columns_and_weights: I,
    ) {
        let (len, _) = columns_and_weights.size_hint();
        self.col_indices.reserve(len);
        for (col, weight) in columns_and_weights {
            self.col_indices.push(col.0.try_into().unwrap());
            self.weights.push(weight);
        }
        self.row_starts.push(self.col_indices.len().try_into().unwrap());
    }
    pub fn is_empty(&self) -> bool {
        self.row_starts.len() <= 1
    }
    pub fn add_to_raw(&self, raw: &mut crate::raw::Model, sos_type: SOSConstraintType) {
        if !self.is_empty() {
            raw.add_sos(&self.row_starts, &self.col_indices, &self.weights, sos_type);
        }
    }
}

impl Default for SOSConstraints {
    fn default() -> Self {
        SOSConstraints {
            row_starts: vec![0],
            col_indices: vec![],
            weights: vec![],
        }
    }
}