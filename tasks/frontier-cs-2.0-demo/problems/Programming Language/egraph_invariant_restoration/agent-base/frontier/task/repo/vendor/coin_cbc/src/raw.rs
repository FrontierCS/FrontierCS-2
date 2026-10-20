//! A raw and safe binding to the Coin CBC C API.
//!
//! The method are as raw as possible to the original API.
//! Differences are:
//!  - snake case naming
//!  - slices as inputs
//!  - rust naming convension (in particular, getter do not begin with `get`)
//!  - assert are used to validate data
//!  - use rust types when cheap (as usize for array length)

use coin_cbc_sys::*;
use std::convert::TryInto;
use std::ffi::CStr;
use std::os::raw::c_int;

#[cfg(feature = "singlethread-cbc")]
fn lock<T, F: FnOnce() -> T>(f: F) -> T {
    lazy_static::lazy_static! {
        static ref GLOBAL_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
    }
    let _lock = GLOBAL_MUTEX.lock().unwrap();
    f()
}

#[cfg(not(feature = "singlethread-cbc"))]
fn lock<T, F: FnOnce() -> T>(f: F) -> T {
    f()
}

/// Sense of optimization.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Sense {
    /// Objective must be minimized.
    Minimize,
    /// Objective must be maximized.
    Maximize,
    /// The objective is ignored, only searching for a feasible
    /// solution.
    Ignore,
}
impl Default for Sense {
    fn default() -> Self {
        Sense::Ignore
    }
}

/// Status of the model.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Status {
    /// The solving procedure was not launched.
    Unlaunched = -1,
    /// The solving procedure finished.
    Finished = 0,
    /// The solving procedure was stopped before optimality was proved.
    Stopped = 1,
    /// The solving procedure was abandoned.
    Abandoned = 2,
    /// The solving procedure is inside a user event.
    UserEvent = 5,
}

#[allow(missing_docs)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SecondaryStatus {
    Unlaunched = -1,
    HasSolution = 0,
    LinearRelaxationInfeasible = 1,
    StoppedOnGap = 2,
    StoppedOnNodes = 3,
    StoppedOnTime = 4,
    StoppedOnUserEvent = 5,
    StoppedOnSolutions = 6,
    LinearRelaxationUnbounded = 7,
    StoppedOnIterationLimit = 8,
}

/// The type of a special ordered set constraint
#[repr(i32)]
pub enum SOSConstraintType {
    /// type 1: at most one element in the given set of columns can be non-zero
    Type1 = 1,
    /// type 2: at most two elements in the given set of columns can be non-zero.
    /// If two elements are non-zero, then they have to be consecutive.
    Type2 = 2,
}

/// A CBC MILP model.
///
/// Their methods are a direct translation from the C API. For
/// documentation, see the official API documentation.
pub struct Model {
    m: *mut Cbc_Model,
}

#[allow(missing_docs)]
impl Model {
    pub fn new() -> Self {
        Self {
            m: lock(|| unsafe { Cbc_newModel() }),
        }
    }
    pub fn version() -> &'static str {
        unsafe { CStr::from_ptr(Cbc_getVersion()).to_str().unwrap() }
    }
    pub fn load_problem(
        &mut self,
        numcols: usize,
        numrows: usize,
        start: &[c_int],
        index: &[c_int],
        value: &[f64],
        collb: Option<&[f64]>,
        colub: Option<&[f64]>,
        obj: Option<&[f64]>,
        rowlb: Option<&[f64]>,
        rowub: Option<&[f64]>,
    ) {
        assert_eq!(start.len(), numcols.checked_add(1).unwrap());
        assert_eq!(index.len(), start[numcols].try_into().unwrap());
        assert!(start[0] >= 0);
        assert!(start.windows(2).all(|w| w[0] <= w[1]
            && index[w[0].try_into().unwrap()..w[1].try_into().unwrap()]
                .windows(2)
                .all(|w| w[0] <= w[1])));

        assert!(collb.map_or(true, |v| v.len() == numcols));
        assert!(colub.map_or(true, |v| v.len() == numcols));
        assert!(obj.map_or(true, |v| v.len() == numcols));
        assert!(rowlb.map_or(true, |v| v.len() == numrows));
        assert!(rowlb.map_or(true, |v| v.len() == numrows));

        fn as_ptr(v: Option<&[f64]>) -> *const f64 {
            match v {
                None => std::ptr::null(),
                Some(v) => v.as_ptr(),
            }
        }

        unsafe {
            Cbc_loadProblem(
                self.m,
                numcols.try_into().unwrap(),
                numrows.try_into().unwrap(),
                start.as_ptr(),
                index.as_ptr(),
                value.as_ptr(),
                as_ptr(collb),
                as_ptr(colub),
                as_ptr(obj),
                as_ptr(rowlb),
                as_ptr(rowub),
            )
        };
    }
    pub fn read_mps(&mut self, filename: &CStr) {
        unsafe { Cbc_readMps(self.m, filename.as_ptr()) };
    }
    pub fn write_mps(&self, filename: &CStr) {
        unsafe { Cbc_writeMps(self.m, filename.as_ptr()) };
    }
    pub fn set_initial_solution(&mut self, sol: &[f64]) {
        assert_eq!(self.num_cols(), sol.len());
        unsafe { Cbc_setInitialSolution(self.m, sol.as_ptr()) };
    }
    // TODO: setProblemName
    pub fn num_elements(&self) -> usize {
        unsafe { Cbc_getNumElements(self.m).try_into().unwrap() }
    }
    pub fn vector_starts(&self) -> &[c_int] {
        unsafe {
            std::slice::from_raw_parts(
                Cbc_getVectorStarts(self.m),
                self.num_cols().checked_add(1).unwrap(),
            )
        }
    }
    pub fn indices(&self) -> &[c_int] {
        let size = (*self.vector_starts().last().unwrap()).try_into().unwrap();
        unsafe { std::slice::from_raw_parts(Cbc_getIndices(self.m), size) }
    }
    pub fn elements(&self) -> &[f64] {
        let size = (*self.vector_starts().last().unwrap()).try_into().unwrap();
        unsafe { std::slice::from_raw_parts(Cbc_getElements(self.m), size) }
    }
    pub fn max_name_length(&self) -> usize {
        unsafe { Cbc_maxNameLength(self.m).try_into().unwrap() }
    }
    // TODO: name management
    pub fn num_rows(&self) -> usize {
        unsafe { Cbc_getNumRows(self.m).try_into().unwrap() }
    }
    pub fn num_cols(&self) -> usize {
        unsafe { Cbc_getNumCols(self.m).try_into().unwrap() }
    }
    pub fn set_obj_sense(&mut self, sense: Sense) {
        let sense = match sense {
            Sense::Minimize => 1.,
            Sense::Maximize => -1.,
            Sense::Ignore => 0.,
        };
        unsafe { Cbc_setObjSense(self.m, sense) };
    }
    pub fn obj_sense(&self) -> Sense {
        let sense = unsafe { Cbc_getObjSense(self.m) };
        if sense == 1. {
            Sense::Minimize
        } else if sense == -1. {
            Sense::Maximize
        } else {
            Sense::Ignore
        }
    }
    pub fn row_lower(&self) -> &[f64] {
        let size = self.num_rows();
        unsafe { std::slice::from_raw_parts(Cbc_getRowLower(self.m), size) }
    }
    pub fn set_row_lower(&mut self, i: usize, value: f64) {
        assert!(i < self.num_rows());
        unsafe { Cbc_setRowLower(self.m, i.try_into().unwrap(), value) }
    }
    pub fn row_upper(&self) -> &[f64] {
        let size = self.num_rows();
        unsafe { std::slice::from_raw_parts(Cbc_getRowUpper(self.m), size) }
    }
    pub fn set_row_upper(&mut self, i: usize, value: f64) {
        assert!(i < self.num_rows());
        unsafe { Cbc_setRowUpper(self.m, i.try_into().unwrap(), value) }
    }
    pub fn obj_coefficients(&self) -> &[f64] {
        let size = self.num_cols();
        unsafe { std::slice::from_raw_parts(Cbc_getObjCoefficients(self.m), size) }
    }
    pub fn set_obj_coeff(&mut self, i: usize, value: f64) {
        assert!(i < self.num_cols());
        unsafe { Cbc_setObjCoeff(self.m, i.try_into().unwrap(), value) }
    }
    pub fn col_lower(&self) -> &[f64] {
        let size = self.num_cols();
        unsafe { std::slice::from_raw_parts(Cbc_getColLower(self.m), size) }
    }
    pub fn set_col_lower(&mut self, i: usize, value: f64) {
        assert!(i < self.num_cols());
        unsafe { Cbc_setColLower(self.m, i.try_into().unwrap(), value) }
    }
    pub fn col_upper(&self) -> &[f64] {
        let size = self.num_cols();
        unsafe { std::slice::from_raw_parts(Cbc_getColUpper(self.m), size) }
    }
    pub fn set_col_upper(&mut self, i: usize, value: f64) {
        assert!(i < self.num_cols());
        unsafe { Cbc_setColUpper(self.m, i.try_into().unwrap(), value) }
    }
    pub fn is_integer(&self, i: usize) -> bool {
        assert!(i < self.num_cols());
        unsafe { Cbc_isInteger(self.m, i.try_into().unwrap()) != 0 }
    }
    pub fn set_continuous(&mut self, i: usize) {
        assert!(i < self.num_cols());
        unsafe { Cbc_setContinuous(self.m, i.try_into().unwrap()) }
    }
    pub fn set_integer(&mut self, i: usize) {
        assert!(i < self.num_cols());
        unsafe { Cbc_setInteger(self.m, i.try_into().unwrap()) }
    }
    /// Adds multiple SOS constraints
    /// num_rows: the number of SOS constraints to add
    /// row_starts: The indices at which each new constraint starts in the col_indices array,
    /// plus one last element that indicates the size of col_indices array.
    /// col_indices: The index of each variable to include in the constraints.
    /// You create this array by concatenating the indices of the columns in each constraint.
    pub fn add_sos(
        &mut self,
        row_starts: &[c_int],
        col_indices: &[c_int],
        weights: &[f64],
        sos_type: SOSConstraintType,
    ) {
        let num_rows = row_starts.len().checked_sub(1).unwrap();
        let last_idx: usize = row_starts[num_rows].try_into().unwrap();
        assert_eq!(last_idx, col_indices.len());
        for starts in row_starts.windows(2) {
            assert!(starts[0] <= starts[1]);
            let idx: usize = starts[0].try_into().unwrap();
            assert!(idx < weights.len());
            let col_idx: usize = col_indices[idx].try_into().unwrap();
            assert!(col_idx <= self.num_cols());
        }
        unsafe {
            Cbc_addSOS(
                self.m,
                num_rows.try_into().unwrap(),
                row_starts.as_ptr(),
                col_indices.as_ptr(),
                weights.as_ptr(),
                sos_type as c_int,
            )
        }
    }
    pub fn print_model(&self, arg_prefix: &CStr) {
        unsafe { Cbc_printModel(self.m, arg_prefix.as_ptr()) }
    }
    pub fn set_parameter(&mut self, name: &CStr, value: &CStr) {
        unsafe { Cbc_setParameter(self.m, name.as_ptr(), value.as_ptr()) };
    }
    pub fn set_log_level(&mut self, log_level: c_int) {
        unsafe { Cbc_setLogLevel(self.m, log_level) };
    }
    // TODO: callback
    pub fn solve(&mut self) -> c_int {
        lock(|| unsafe { Cbc_solve(self.m) })
    }
    pub fn sum_primal_infeasibilities(&self) -> f64 {
        unsafe { Cbc_sumPrimalInfeasibilities(self.m) }
    }
    pub fn number_primal_infeasibilities(&self) -> c_int {
        unsafe { Cbc_numberPrimalInfeasibilities(self.m) }
    }
    pub fn check_solution(&mut self) {
        unsafe { Cbc_checkSolution(self.m) }
    }
    pub fn iteration_count(&self) -> c_int {
        unsafe { Cbc_getIterationCount(self.m) }
    }
    pub fn is_abandoned(&self) -> bool {
        unsafe { Cbc_isAbandoned(self.m) != 0 }
    }
    pub fn is_proven_optimal(&self) -> bool {
        unsafe { Cbc_isProvenOptimal(self.m) != 0 }
    }
    pub fn is_proven_infeasible(&self) -> bool {
        unsafe { Cbc_isProvenInfeasible(self.m) != 0 }
    }
    pub fn is_continuous_unbounded(&self) -> bool {
        unsafe { Cbc_isContinuousUnbounded(self.m) != 0 }
    }
    pub fn is_node_limit_reached(&self) -> bool {
        unsafe { Cbc_isNodeLimitReached(self.m) != 0 }
    }
    pub fn is_seconds_limit_reached(&self) -> bool {
        unsafe { Cbc_isSecondsLimitReached(self.m) != 0 }
    }
    pub fn is_solution_limit_reached(&self) -> bool {
        unsafe { Cbc_isSolutionLimitReached(self.m) != 0 }
    }
    pub fn is_initial_solve_abandoned(&self) -> bool {
        unsafe { Cbc_isInitialSolveAbandoned(self.m) != 0 }
    }
    pub fn is_initial_solve_proven_optimal(&self) -> bool {
        unsafe { Cbc_isInitialSolveProvenOptimal(self.m) != 0 }
    }
    pub fn is_initial_solve_proven_primal_infeasible(&self) -> bool {
        unsafe { Cbc_isInitialSolveProvenPrimalInfeasible(self.m) != 0 }
    }

    #[cfg(feature = "cbc-310")]
    #[cfg_attr(docsrs, doc(cfg(feature = "cbc-310")))]
    /// Primal row solution
    /// This function is not available on libcbc < 3.10
    pub fn row_activity(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(Cbc_getRowActivity(self.m), self.num_rows()) }
    }

    /// Primal column solution
    pub fn col_solution(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(Cbc_getColSolution(self.m), self.num_cols()) }
    }
    /* Disabled until available in the C API
    /// Dual row solution
    pub fn row_price(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(Cbc_getRowPrice(self.m), self.num_rows()) }
    }
     */

    #[cfg(feature = "cbc-310")]
    #[cfg_attr(docsrs, doc(cfg(feature = "cbc-310")))]
    /// Dual column solution.
    /// This function is not available on libcbc < 3.10.
    pub fn reduced_cost(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(Cbc_getReducedCost(self.m), self.num_cols()) }
    }
    pub fn obj_value(&self) -> f64 {
        unsafe { Cbc_getObjValue(self.m) }
    }
    pub fn best_possible_value(&self) -> f64 {
        unsafe { Cbc_getBestPossibleObjValue(self.m) }
    }
    pub fn print_solution(&self) {
        unsafe { Cbc_printSolution(self.m) }
    }
    pub fn status(&self) -> Status {
        match unsafe { Cbc_status(self.m) } {
            s if s == Status::Unlaunched as c_int => Status::Unlaunched,
            s if s == Status::Finished as c_int => Status::Finished,
            s if s == Status::Stopped as c_int => Status::Stopped,
            s if s == Status::Abandoned as c_int => Status::Abandoned,
            s if s == Status::UserEvent as c_int => Status::UserEvent,
            _ => unreachable!(),
        }
    }
    pub fn secondary_status(&self) -> SecondaryStatus {
        use SecondaryStatus::*;
        match unsafe { Cbc_secondaryStatus(self.m) } {
            s if s == Unlaunched as c_int => Unlaunched,
            s if s == HasSolution as c_int => HasSolution,
            s if s == LinearRelaxationInfeasible as c_int => LinearRelaxationInfeasible,
            s if s == StoppedOnGap as c_int => StoppedOnGap,
            s if s == StoppedOnNodes as c_int => StoppedOnNodes,
            s if s == StoppedOnTime as c_int => StoppedOnTime,
            s if s == StoppedOnUserEvent as c_int => StoppedOnUserEvent,
            s if s == StoppedOnSolutions as c_int => StoppedOnSolutions,
            s if s == LinearRelaxationUnbounded as c_int => LinearRelaxationUnbounded,
            s if s == StoppedOnIterationLimit as c_int => StoppedOnIterationLimit,
            _ => unreachable!(),
        }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        lock(|| unsafe { Cbc_deleteModel(self.m) })
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Model {
    fn clone(&self) -> Self {
        Self {
            m: lock(|| unsafe { Cbc_clone(self.m) }),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn knapsack() {
        let mut m = Model::new();
        assert!(Model::version().len() > 4);
        m.load_problem(
            5,
            1,
            &vec![0, 1, 2, 3, 4, 5],
            &vec![0, 0, 0, 0, 0],
            &vec![2., 8., 4., 2., 5.],
            Some(&vec![0., 0., 0., 0., 0.]),
            Some(&vec![1., 1., 1., 1., 1.]),
            Some(&vec![5., 3., 2., 7., 4.]),
            Some(&vec![-std::f64::INFINITY]),
            Some(&vec![10.]),
        );
        assert_eq!(5, m.num_cols());
        assert_eq!(1, m.num_rows());
        m.set_obj_sense(Sense::Maximize);
        assert_eq!(Sense::Maximize, m.obj_sense());
        for i in 0..5 {
            m.set_integer(i);
            assert!(m.is_integer(i));
        }
        m.set_initial_solution(&vec![1., 1., 0., 0., 0.]);
        m.solve();
        assert_eq!(Status::Finished, m.status());
        assert!(m.is_proven_optimal());
        assert!(!m.is_abandoned());
        assert!(!m.is_proven_infeasible());
        assert!(!m.is_continuous_unbounded());
        assert!(!m.is_node_limit_reached());
        assert!(!m.is_seconds_limit_reached());
        assert!(!m.is_solution_limit_reached());
        assert!((m.obj_value() - 16.).abs() < 1e-6);
        assert!((m.best_possible_value() - 16.).abs() < 1e-6);
        let sol = m.col_solution();
        assert!((sol[0] - 1.).abs() < 1e-6);
        assert!((sol[1] - 0.).abs() < 1e-6);
        assert!((sol[2] - 0.).abs() < 1e-6);
        assert!((sol[3] - 1.).abs() < 1e-6);
        assert!((sol[4] - 1.).abs() < 1e-6);
    }

    #[test]
    fn big_row() {
        let mut m = Model::new();
        let numcols = 0;
        let numrows = 1_000;
        let start = [1];
        let value = [0.];
        m.load_problem(
            numcols, numrows, &start, &start, &value, None, None, None, None, None,
        );
        m.set_initial_solution(&[]);
        assert_eq!(&value, &[0.])
    }

    #[test]
    fn multiple_threads() {
        use std::thread::{spawn, JoinHandle};
        let threads: Vec<JoinHandle<()>> = (0..50)
            .map(|_| {
                spawn(|| {
                    for _ in 1..3 {
                        // Solve an empty problem
                        let mut m = Model::new();
                        m.load_problem(1, 0, &[0, 0], &[], &[], None, None, None, None, None);
                        m.solve();
                        if Model::version().starts_with("2.9.") {
                            assert_eq!(Status::Finished, m.status());
                            assert_eq!(SecondaryStatus::HasSolution, m.secondary_status());
                        } else {
                            assert_eq!(Status::Unlaunched, m.status());
                            assert_eq!(SecondaryStatus::Unlaunched, m.secondary_status());
                        }
                        assert!((m.col_solution()[0]).abs() < 1e-6);
                        // Solve a non-empty problem
                        let mut m = Model::new();
                        m.load_problem(
                            5,
                            1,
                            &vec![0, 1, 2, 3, 4, 5],
                            &vec![0, 0, 0, 0, 0],
                            &vec![2., 8., 4., 2., 5.],
                            Some(&vec![0., 0., 0., 0., 0.]),
                            Some(&vec![1., 1., 1., 1., 1.]),
                            Some(&vec![5., 3., 2., 7., 4.]),
                            Some(&vec![-std::f64::INFINITY]),
                            Some(&vec![10.]),
                        );
                        m.set_obj_sense(Sense::Maximize);
                        for i in 0..5 {
                            m.set_integer(i);
                        }
                        m.set_initial_solution(&vec![1., 1., 0., 0., 0.]);
                        m.solve();
                        assert_eq!(Status::Finished, m.status());
                        assert_eq!(SecondaryStatus::HasSolution, m.secondary_status());
                        assert!((m.col_solution()[0] - 1.).abs() < 1e-6);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().expect("thread failed");
        }
    }

    #[test]
    fn sos_one_constraint() {
        let mut m = Model::new();
        // Minimize 5x + 3y with -1 <= x <= 1 and -1 <= y <= 1
        m.load_problem(
            2,
            0,
            &vec![0, 0, 0],
            &vec![],
            &vec![],
            Some(&vec![-1., -1.]),
            Some(&vec![1., 1.]),
            Some(&vec![5., 3.]),
            None,
            None,
        );
        // Add a constraint that either x or y must be 0
        m.add_sos(&[0, 2], &[0, 1], &[5., 3.], SOSConstraintType::Type1);
        m.set_integer(0);
        m.set_integer(1);
        m.solve();
        // The solution is x = -1 and y = 0
        assert_eq!(&[-1., 0.], m.col_solution());
    }

    #[test]
    fn sos_multiple_constraints() {
        let mut m = Model::new();
        // Minimize x + 5y + z with -1 <= x <= 1 and -1 <= y <= 1 and -1 <= z <= 1
        m.load_problem(
            3,
            0,
            &vec![0, 0, 0, 0],
            &vec![],
            &vec![],
            Some(&vec![-1., -1., -1.]),
            Some(&vec![1., 1., 1.]),
            Some(&vec![1., 5., 1.]),
            None,
            None,
        );
        // Add a constraint that either x or y must be 0
        m.add_sos(
            &[
                0, 2, // The first constraint is on columns col_indices[0..2]
                4, // The second is on columns col_indices[2..4]
            ],
            &[
                0, 1, // The first constraint is that either x or y must be 0
                1, 2, // The second constraint is that either y or z must be 0
            ],
            &[1., 5., 5., 1.],
            SOSConstraintType::Type1,
        );
        m.set_integer(0);
        m.set_integer(1);
        m.set_integer(2);

        m.solve();
        // The solution is x = -1 and y = 0
        assert_eq!(&[0., -1., 0.], m.col_solution());
    }
}
