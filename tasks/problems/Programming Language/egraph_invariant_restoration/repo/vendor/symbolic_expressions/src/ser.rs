// (c) 2016-2017 Productize SPRL <joost@productize.be>

use std::io;

use formatter::*;

use Sexp;
use error::SexpError;
use encode_string;

struct Serializer<W, F = CompactFormatter> {
    writer: W,
    formatter: F,
}

// dispatches only based on Formatter
impl<W> Serializer<W>
where
    W: io::Write,
{
    fn new(writer: W) -> Self {
        Serializer::with_formatter(writer, CompactFormatter)
    }
}

impl<W> Serializer<W, RulesFormatter>
where
    W: io::Write,
{
    fn new_rules(writer: W, rules: Rules) -> Self {
        Serializer::with_formatter(writer, RulesFormatter::new(rules))
    }
}

impl<W, F> Serializer<W, F>
where
    W: io::Write,
    F: Formatter,
{
    fn with_formatter(writer: W, formatter: F) -> Self {
        Serializer {
            writer: writer,
            formatter: formatter,
        }
    }

    fn serialize_str(&mut self, value: &str) -> Result<(), SexpError> {
        write!(&mut self.writer, "{}", encode_string(value)).map_err(From::from)
    }

    fn serialize(&mut self, value: &Sexp) -> Result<(), SexpError> {
        match *value {
            Sexp::String(ref s) => self.serialize_str(s),
            Sexp::List(ref list) => {
                let mut first = true;
                if list.is_empty() {
                    self.formatter.open(&mut self.writer, None)?;
                } else {
                    for v in list {
                        if first {
                            self.formatter.open(&mut self.writer, Some(v))?;
                            self.serialize(v)?;
                            first = false;
                        } else {
                            self.formatter.element(&mut self.writer, v)?;
                            self.serialize(v)?;
                        }
                    }
                }
                self.formatter.close(&mut self.writer)
            }
            Sexp::Empty => Ok(()),
        }
    }
}

/// serialize a symbolic-expression to a Writer
pub fn to_writer<W>(writer: &mut W, value: &Sexp) -> Result<(), SexpError>
where
    W: io::Write,
{
    let mut ser = Serializer::new(writer);
    ser.serialize(value)
}

/// serialize a symbolic-expression to a Writer using a Formatter
pub fn to_writer_with_formatter<W, F>(
    writer: &mut W,
    formatter: F,
    value: &Sexp,
) -> Result<(), SexpError>
where
    W: io::Write,
    F: Formatter,
{
    let mut ser = Serializer::with_formatter(writer, formatter);
    ser.serialize(value)
}

/// serialize a symbolic-expression to a Writer using a Rules Formatter
pub fn to_writer_with_rules<W>(writer: &mut W, rules: Rules, value: &Sexp) -> Result<(), SexpError>
where
    W: io::Write,
{
    let mut ser = Serializer::new_rules(writer, rules);
    ser.serialize(value)
}

/// serialize a symbolic-expression to a Vec<u8>
pub fn to_vec(value: &Sexp) -> Result<Vec<u8>, SexpError> {
    let mut writer = Vec::with_capacity(128);
    to_writer(&mut writer, value)?;
    Ok(writer)
}

/// serialize a symbolic-expression to a Vec<u8> using Rules
pub fn to_vec_with_rules(value: &Sexp, rules: Rules) -> Result<Vec<u8>, SexpError> {
    let mut writer = Vec::with_capacity(128);
    to_writer_with_rules(&mut writer, rules, value)?;
    Ok(writer)
}

/// serialize a symbolic-expression to a Vec<u8> using a Formatter
pub fn to_vec_with_formatter<F>(value: &Sexp, formatter: F) -> Result<Vec<u8>, SexpError>
where
    F: Formatter,
{
    let mut writer = Vec::with_capacity(128);
    to_writer_with_formatter(&mut writer, formatter, value)?;
    Ok(writer)
}

/// serialize a symbolic-expression to a String
pub fn to_string(value: &Sexp) -> Result<String, SexpError> {
    let vec = to_vec(value)?;
    let string = String::from_utf8(vec)?;
    Ok(string)
}

/// serialize a symbolic-expression to a String using Rules
pub fn to_string_with_rules(value: &Sexp, rules: Rules) -> Result<String, SexpError> {
    let vec = to_vec_with_rules(value, rules)?;
    let string = String::from_utf8(vec)?;
    Ok(string)
}

/// serialize a symbolic-expression to a String using a Formatter
pub fn to_string_with_formatter<F>(value: &Sexp, formatter: F) -> Result<String, SexpError>
where
    F: Formatter,
{
    let vec = to_vec_with_formatter(value, formatter)?;
    let string = String::from_utf8(vec)?;
    Ok(string)
}
