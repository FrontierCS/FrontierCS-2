//! Raw coin cbc bindings for CBC 2.9. For documentation see official
//! documentation.
#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int};

#[repr(C)]
pub struct Cbc_Model {
    _private: [u8; 0],
}
pub type CoinBigIndex = c_int;
pub type cbc_callback = Option<
    unsafe extern "C" fn(
        model: *mut Cbc_Model,
        msgno: c_int,
        ndouble: c_int,
        dvec: *const f64,
        nint: c_int,
        ivec: *const c_int,
        nchar: c_int,
        cvec: *mut *mut c_char,
    ),
>;

#[link(name = "CbcSolver")]
extern "C" {
    pub fn Cbc_newModel() -> *mut Cbc_Model;
    pub fn Cbc_deleteModel(model: *mut Cbc_Model);
    pub fn Cbc_getVersion() -> *const c_char;
    pub fn Cbc_loadProblem(
        model: *mut Cbc_Model,
        numcols: c_int,
        numrows: c_int,
        start: *const CoinBigIndex,
        index: *const c_int,
        value: *const f64,
        collb: *const f64,
        colub: *const f64,
        obj: *const f64,
        rowlb: *const f64,
        rowub: *const f64,
    );
    pub fn Cbc_readMps(model: *mut Cbc_Model, filename: *const c_char) -> c_int;
    pub fn Cbc_writeMps(model: *mut Cbc_Model, filename: *const c_char);
    pub fn Cbc_setInitialSolution(model: *mut Cbc_Model, sol: *const f64);
    pub fn Cbc_problemName(model: *mut Cbc_Model, maxNumberCharacters: c_int, array: *mut c_char);
    pub fn Cbc_setProblemName(model: *mut Cbc_Model, array: *const c_char) -> c_int;
    pub fn Cbc_getNumElements(model: *mut Cbc_Model) -> c_int;
    pub fn Cbc_getVectorStarts(model: *mut Cbc_Model) -> *const CoinBigIndex;
    pub fn Cbc_getIndices(model: *mut Cbc_Model) -> *const c_int;
    pub fn Cbc_getElements(model: *mut Cbc_Model) -> *const f64;
    pub fn Cbc_maxNameLength(model: *mut Cbc_Model) -> usize;
    pub fn Cbc_getRowName(model: *mut Cbc_Model, iRow: c_int, name: *mut c_char, maxLength: usize);
    pub fn Cbc_getColName(
        model: *mut Cbc_Model,
        iColumn: c_int,
        name: *mut c_char,
        maxLength: usize,
    );
    pub fn Cbc_setColName(model: *mut Cbc_Model, iColumn: c_int, name: *const c_char);
    pub fn Cbc_setRowName(model: *mut Cbc_Model, iRow: c_int, name: *const c_char);
    pub fn Cbc_getNumRows(model: *mut Cbc_Model) -> c_int;
    pub fn Cbc_getNumCols(model: *mut Cbc_Model) -> c_int;
    pub fn Cbc_setObjSense(model: *mut Cbc_Model, sense: f64);
    pub fn Cbc_getObjSense(model: *mut Cbc_Model) -> f64;
    pub fn Cbc_getRowLower(model: *mut Cbc_Model) -> *const f64;
    pub fn Cbc_setRowLower(model: *mut Cbc_Model, index: c_int, value: f64);
    pub fn Cbc_getRowUpper(model: *mut Cbc_Model) -> *const f64;
    pub fn Cbc_setRowUpper(model: *mut Cbc_Model, index: c_int, value: f64);
    pub fn Cbc_getObjCoefficients(model: *mut Cbc_Model) -> *const f64;
    pub fn Cbc_setObjCoeff(model: *mut Cbc_Model, index: c_int, value: f64);
    pub fn Cbc_getColLower(model: *mut Cbc_Model) -> *const f64;
    pub fn Cbc_setColLower(model: *mut Cbc_Model, index: c_int, value: f64);
    pub fn Cbc_getColUpper(model: *mut Cbc_Model) -> *const f64;
    pub fn Cbc_setColUpper(model: *mut Cbc_Model, index: c_int, value: f64);
    pub fn Cbc_isInteger(model: *mut Cbc_Model, i: c_int) -> c_int;
    pub fn Cbc_setContinuous(model: *mut Cbc_Model, iColumn: c_int);
    pub fn Cbc_setInteger(model: *mut Cbc_Model, iColumn: c_int);
    pub fn Cbc_addSOS(
        model: *mut Cbc_Model,
        numRows: c_int,
        rowStarts: *const c_int,
        colIndices: *const c_int,
        weights: *const f64,
        type_: c_int,
    );
    pub fn Cbc_printModel(model: *mut Cbc_Model, argPrefix: *const c_char);
    pub fn Cbc_clone(model: *mut Cbc_Model) -> *mut Cbc_Model;
    pub fn Cbc_setParameter(model: *mut Cbc_Model, name: *const c_char, value: *const c_char);
    pub fn Cbc_setLogLevel(model: *mut Cbc_Model, logLevel: c_int);
    pub fn Cbc_registerCallBack(model: *mut Cbc_Model, userCallBack: cbc_callback);
    pub fn Cbc_clearCallBack(model: *mut Cbc_Model);
    pub fn Cbc_solve(model: *mut Cbc_Model) -> c_int;
    pub fn Cbc_sumPrimalInfeasibilities(model: *mut Cbc_Model) -> f64;
    pub fn Cbc_numberPrimalInfeasibilities(model: *mut Cbc_Model) -> c_int;
    pub fn Cbc_checkSolution(model: *mut Cbc_Model);
    pub fn Cbc_getIterationCount(model: *mut Cbc_Model) -> c_int;
    pub fn Cbc_isAbandoned(model: *mut Cbc_Model) -> c_int;
    pub fn Cbc_isProvenOptimal(model: *mut Cbc_Model) -> c_int;
    pub fn Cbc_isProvenInfeasible(model: *mut Cbc_Model) -> c_int;
    pub fn Cbc_isContinuousUnbounded(model: *mut Cbc_Model) -> c_int;
    pub fn Cbc_isNodeLimitReached(model: *mut Cbc_Model) -> c_int;
    pub fn Cbc_isSecondsLimitReached(model: *mut Cbc_Model) -> c_int;
    pub fn Cbc_isSolutionLimitReached(model: *mut Cbc_Model) -> c_int;
    pub fn Cbc_isInitialSolveAbandoned(model: *mut Cbc_Model) -> c_int;
    pub fn Cbc_isInitialSolveProvenOptimal(model: *mut Cbc_Model) -> c_int;
    pub fn Cbc_isInitialSolveProvenPrimalInfeasible(model: *mut Cbc_Model) -> c_int;
    /// Primal row solution
    pub fn Cbc_getRowActivity(model: *mut Cbc_Model) -> *const f64;
    /// Primal column solution
    pub fn Cbc_getColSolution(model: *mut Cbc_Model) -> *const f64;
    /* Disabled until available in the C API: https://github.com/coin-or/Cbc/issues/370
    /// Dual row solution
    pub fn Cbc_getRowPrice(model: *mut Cbc_Model) -> *const f64;
     */
    /// Dual column solution
    pub fn Cbc_getReducedCost(model: *mut Cbc_Model) -> *const f64;
    pub fn Cbc_getObjValue(model: *mut Cbc_Model) -> f64;
    pub fn Cbc_getBestPossibleObjValue(model: *mut Cbc_Model) -> f64;
    pub fn Cbc_getNodeCount(model: *mut Cbc_Model) -> c_int;
    pub fn Cbc_printSolution(model: *mut Cbc_Model);
    pub fn Cbc_status(model: *mut Cbc_Model) -> c_int;
    pub fn Cbc_secondaryStatus(model: *mut Cbc_Model) -> c_int;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knapsack() {
        // Simple knapsack problem
        // Maximize  5x[1] + 3x[2] + 2x[3] + 7x[4] + 4x[5]
        // s.t.      2x[1] + 8x[2] + 4x[3] + 2x[4] + 5x[5] <= 10
        // All x binary
        let start = [0, 1, 2, 3, 4, 5];
        let rowindex = [0, 0, 0, 0, 0];
        let value = [2., 8., 4., 2., 5.];
        let collb = [0., 0., 0., 0., 0.];
        let colub = [1., 1., 1., 1., 1.];
        let obj = [5., 3., 2., 7., 4.];
        let feasible = [1., 1., 0., 0., 0.];
        let rowlb = [-std::f64::INFINITY];
        let rowub = [10.];

        unsafe {
            let model = Cbc_newModel();
            Cbc_loadProblem(
                model,
                5,
                1,
                start.as_ptr(),
                rowindex.as_ptr(),
                value.as_ptr(),
                collb.as_ptr(),
                colub.as_ptr(),
                obj.as_ptr(),
                rowlb.as_ptr(),
                rowub.as_ptr(),
            );
            assert_eq!(5, Cbc_getNumCols(model));
            assert_eq!(1, Cbc_getNumRows(model));
            for i in 0..5 {
                Cbc_setInteger(model, i);
                assert!(Cbc_isInteger(model, i) != 0);
            }
            Cbc_setObjSense(model, -1.);
            assert_eq!(-1., Cbc_getObjSense(model));
            Cbc_setInitialSolution(model, feasible.as_ptr());

            Cbc_solve(model);

            assert!(Cbc_isProvenOptimal(model) != 0);
            assert!((Cbc_getObjValue(model) - 16.).abs() < 1e-6);
            assert!((Cbc_getBestPossibleObjValue(model) - 16.).abs() < 1e-6);

            let sol = std::slice::from_raw_parts(Cbc_getColSolution(model), 5);
            assert!((sol[0] - 1.).abs() < 1e-6);
            assert!((sol[1] - 0.).abs() < 1e-6);
            assert!((sol[2] - 0.).abs() < 1e-6);
            assert!((sol[3] - 1.).abs() < 1e-6);
            assert!((sol[4] - 1.).abs() < 1e-6);

            Cbc_deleteModel(model);
        }
    }
}
