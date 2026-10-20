// (c) 2016 Productize SPRL <joost@productize.be>

use std::io;
use std::collections::HashMap;

use error::SexpError;
use Sexp;

/// trait for formatting the serialization of a symbolic-expression
pub trait Formatter {
    /// Called when serializing a '('.
    fn open<W>(&mut self, writer: &mut W, value: Option<&Sexp>) -> Result<(), SexpError>
    where
        W: io::Write;

    /// Called when serializing a ' VAL'.
    fn element<W>(&mut self, writer: &mut W, value: &Sexp) -> Result<(), SexpError>
    where
        W: io::Write;

    /// Called when serializing a ')'.
    fn close<W>(&mut self, writer: &mut W) -> Result<(), SexpError>
    where
        W: io::Write;
}

pub struct CompactFormatter;

impl Formatter for CompactFormatter {
    fn open<W>(&mut self, writer: &mut W, _value: Option<&Sexp>) -> Result<(), SexpError>
    where
        W: io::Write,
    {
        writer.write_all(b"(").map_err(From::from)
    }
    fn element<W>(&mut self, writer: &mut W, _value: &Sexp) -> Result<(), SexpError>
    where
        W: io::Write,
    {
        writer.write_all(b" ").map_err(From::from)
    }

    fn close<W>(&mut self, writer: &mut W) -> Result<(), SexpError>
    where
        W: io::Write,
    {
        writer.write_all(b")").map_err(From::from)
    }
}

/// datatype used by the example Formatter `RulesFormatter`
pub type Rules = HashMap<&'static str, i64>;

pub struct RulesFormatter {
    indent: Vec<u8>,
    indent_before: Rules,
}

impl Default for RulesFormatter {
    fn default() -> Self {
        let idb = HashMap::new();
        RulesFormatter {
            indent: vec![b' ', b' '], // two spaces
            indent_before: idb,
        }
    }
}

impl RulesFormatter {
    pub fn new(indent_before: Rules) -> RulesFormatter {
        RulesFormatter {
            indent: vec![b' ', b' '], // two spaces
            indent_before: indent_before,
        }
    }
}

impl Formatter for RulesFormatter {
    fn open<W>(&mut self, writer: &mut W, value: Option<&Sexp>) -> Result<(), SexpError>
    where
        W: io::Write,
    {
        // if first element is string and it has an indent setting
        if let Some(sexp) = value {
            if let Sexp::String(ref s) = *sexp {
                let s: &str = s;
                if let Some(&i) = self.indent_before.get(s) {
                    writer.write_all(b"\n")?;
                    for _ in 0..i {
                        writer.write_all(&self.indent)?;
                    }
                }
            }
        }
        writer.write_all(b"(").map_err(From::from)
    }

    fn element<W>(&mut self, writer: &mut W, value: &Sexp) -> Result<(), SexpError>
    where
        W: io::Write,
    {
        // if containing value is a list and it has indent_before
        // don't put the space
        if let Sexp::List(ref l) = *value {
            if !l.is_empty() {
                if let Sexp::String(ref s) = l[0] {
                    let s: &str = s; // why needed?
                    if self.indent_before.contains_key(s) {
                        return Ok(());
                    }
                }
            }
        }
        writer.write_all(b" ").map_err(From::from)
    }

    fn close<W>(&mut self, writer: &mut W) -> Result<(), SexpError>
    where
        W: io::Write,
    {
        writer.write_all(b")").map_err(From::from)
    }
}
