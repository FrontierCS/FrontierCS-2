// (c) 2016-2017 Productize SPRL <joost@productize.be>

use std::string;
use std::io;
use std::num;
use std::error;
use std::fmt;

/// errors that can happen in this library
#[derive(Debug)]
pub enum SexpError {
    /// parse error
    Parse(ParseError),
    /// other error
    Other(String),
    /// IO Error
    Io(io::Error),
    /// Utf8 Error parsing error
    FromUtf8(string::FromUtf8Error),
    /// floating point parsing error
    Float(num::ParseFloatError),
    /// integer parsing error
    Int(num::ParseIntError),
}

// TODO: get rid of this again later
impl fmt::Display for SexpError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            SexpError::Parse(ref pe) => write!(f, "symbolic expression parse error: {:?}", pe),
            SexpError::Other(ref s) => write!(f, "symbolic expression other error: {}", s),
            SexpError::Io(ref e) => e.fmt(f),
            SexpError::FromUtf8(ref e) => e.fmt(f),
            SexpError::Float(ref e) => e.fmt(f),
            SexpError::Int(ref e) => e.fmt(f),
        }
    }
}

// TODO: get rid of this again later
impl error::Error for SexpError {
    fn description(&self) -> &str {
        "symbolic expressions error"
    }

    fn cause(&self) -> Option<&error::Error> {
        match *self {
            SexpError::Parse(_) | SexpError::Other(_) => None,
            SexpError::Io(ref e) => Some(e),
            SexpError::FromUtf8(ref e) => Some(e),
            SexpError::Float(ref e) => Some(e),
            SexpError::Int(ref e) => Some(e),
        }
    }
}

/// detailed symbolic-expression parse error information
#[derive(Debug)]
pub struct ParseError {
    msg: String,
    line: usize,
    col: usize,
}

impl From<io::Error> for SexpError {
    fn from(e: io::Error) -> SexpError {
        SexpError::Io(e)
    }
}

impl From<String> for SexpError {
    fn from(e: String) -> SexpError {
        SexpError::Other(e)
    }
}

impl<'a> From<&'a str> for SexpError {
    fn from(e: &'a str) -> SexpError {
        SexpError::Other(e.into())
    }
}

impl From<string::FromUtf8Error> for SexpError {
    fn from(e: string::FromUtf8Error) -> SexpError {
        SexpError::FromUtf8(e)
    }
}

impl From<num::ParseFloatError> for SexpError {
    fn from(e: num::ParseFloatError) -> SexpError {
        SexpError::Float(e)
    }
}

impl From<num::ParseIntError> for SexpError {
    fn from(e: num::ParseIntError) -> SexpError {
        SexpError::Int(e)
    }
}

/// utility function that creates a symbolic-expressions Error Result for a parser error
pub fn parse_error<T>(line: usize, col: usize, msg: String) -> Result<T, SexpError> {
    let pe = ParseError {
        msg: msg,
        line: line,
        col: col,
    };
    Err(SexpError::Parse(pe))
}
