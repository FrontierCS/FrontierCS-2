// (c) 2015-2017 Productize SPRL <joost@productize.be>

//! symbolic-expressions parsing and generating library

#![warn(missing_docs)]

pub use error::*;
pub use sexp::*;

mod error;
mod formatter;
mod sexp;

/// symbolic-expression parser code: data -> symbolic-expression
pub mod parser;

/// symbolic-expression serialization code: symbolic-expression -> data
pub mod ser;

/// high-level API for deconstructing symbolic-expressions
pub mod iteratom;

pub use formatter::Rules;
pub use formatter::Formatter;

pub use iteratom::from_sexp;

#[cfg(test)]
mod test;
