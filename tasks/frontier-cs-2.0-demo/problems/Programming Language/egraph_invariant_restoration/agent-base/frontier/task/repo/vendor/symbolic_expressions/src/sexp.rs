// (c) 2016-2017 Productize SPRL <joost@productize.be>

use std::fmt;
use std::mem;

use error::SexpError;
/// like Into trait but works from a ref avoiding consumption or expensive clone
pub trait IntoSexp {
    /// convert self into a Sexp
    fn into_sexp(&self) -> Sexp;
}

impl Into<Sexp> for Vec<Sexp> {
    fn into(self: Vec<Sexp>) -> Sexp {
        Sexp::List(self)
    }
}

impl Into<Sexp> for String {
    fn into(self: String) -> Sexp {
        Sexp::String(self)
    }
}

impl<'a> From<&'a str> for Sexp {
    fn from(t: &str) -> Sexp {
        Sexp::String(t.into())
    }
}

impl<'a> From<&'a String> for Sexp {
    fn from(t: &String) -> Sexp {
        Sexp::String(t.clone())
    }
}

impl Into<Sexp> for i64 {
    fn into(self: i64) -> Sexp {
        Sexp::String(format!("{}", self))
    }
}

impl Into<Sexp> for f64 {
    fn into(self: f64) -> Sexp {
        Sexp::String(format!("{}", self))
    }
}

impl<'a> From<(&'a str, Sexp)> for Sexp {
    fn from(kv: (&str, Sexp)) -> Sexp {
        let (name, value) = kv;
        let mut v = vec![];
        v.push(name.into());
        v.push(value);
        v.into()
    }
}

impl<'a, T: fmt::Display> From<(&'a str, &'a T)> for Sexp {
    fn from(kv: (&str, &T)) -> Sexp {
        let (name, value) = kv;
        let mut v = vec![];
        v.push(name.into());
        v.push(format!("{}", value).into());
        v.into()
    }
}

/// a symbolic-expression structure
/// Can be a string or a list or nothing
///
/// `String` shape: hello
/// `List` shape: (...)
/// `Empty shape:
#[derive(Debug, Clone, PartialEq)]
pub enum Sexp {
    /// plain String symbolic-expression
    String(String),
    /// list symbolic-expression
    List(Vec<Sexp>),
    /// empty, trivial symbolic-expression
    Empty,
}

// Following the KiCad file formats specification chapter 4.4 - Identifiers and Strings:
// A string may not contain an actual newline or carriage return,
// but it may use an escape sequence to encode a // newline, such as \n.
// If a string has any of the following conditions, then it must be quoted
// with a leading and trailing double quote
// character, otherwise it is acceptable to not quote the string:
// 1. has one or more of the following 4 separator bytes: ASCII space,
// tab, '(', or ')'.
// 2. has one or more of the following bytes: '%', '{', or '}'.
// 3. has a length of zero bytes, and you need a place holder for the field,
//  then use "".
// 4. includes a byte of '-', and this byte is not in the first position of
// the string.
//

// Joost remark: kicad no longer seems to follow Rule 4.
// so we don't either

/// encode a string according to the guidelines given by Kicad
pub fn encode_string(s: &str) -> String {
    // fn rule_4(s:&str) -> bool {
    // s.contains('-') && s.len() > 1 && s.as_bytes()[0] != 45
    // }
    if s.contains('(') || s.contains(' ') || s.contains(')') || s.contains('\t') || s.contains('{')
        || s.contains('}') || s.contains('}') || s.contains('%') || s.is_empty()
    // || rule_4(s)
    {
        format!("\"{}\"", s)
    } else {
        String::from(s)
    }
}

impl Sexp {
    /// create an empty symbolic-expression
    #[deprecated(since = "4.0.0", note = "please use `Sexp::default()` instead")]
    pub fn new_empty() -> Sexp {
        Sexp::Empty
    }

    /// create a String type symbolic-expression
    #[deprecated(since = "4.0.3", note = "please use `.into()` instead")]
    pub fn new_string<T>(s: T) -> Sexp
    where
        T: fmt::Display,
    {
        format!("{}", s).into()
    }

    /// create a list type symbolic-expression
    #[deprecated(since = "4.0.3", note = "please use `.into()` instead")]
    pub fn new_list(v: Vec<Sexp>) -> Sexp {
        Sexp::List(v)
    }

    /// create an empty list type symbolic-expression
    pub fn start(name: &str) -> Sexp {
        let mut v = vec![];
        v.push(Sexp::String(name.into()));
        Sexp::List(v)
    }

    /// push an element in a list
    pub fn push<T: Into<Sexp>>(&mut self, element: T) {
        match *self {
            Sexp::List(ref mut v) => v.push(element.into()),
            _ => panic!("Only use push on lists!"),
        }
    }

    /// create a list type symbolic-expression where
    /// the first element of the list is a string that indicates
    /// the name, the remainder is filled in via the provided
    /// fill function
    ///
    /// shape: (name ...)
    pub fn new_named_list<F>(name: &str, fill: F) -> Sexp
    where
        F: Fn(&mut Vec<Sexp>),
    {
        let mut v = vec![];
        v.push(name.into());
        fill(&mut v);
        Sexp::List(v)
    }

    /// create a list type symbolic-expression where
    /// the first element of the list is a string that indicates
    /// the name
    #[deprecated(since = "4.0.3", note = "please use `.into()` instead")]
    pub fn new_named<T>(name: &str, value: T) -> Sexp
    where
        T: fmt::Display,
    {
        let mut v = vec![];
        v.push(name.into());
        v.push(format!("{}", value).into());
        v.into()
    }

    /// create a list type symbolic-expression where
    /// the first element of the list is a string that indicates
    /// the name, and the second is another symbolic-expression
    /// created via the IntoSexp trait
    #[deprecated(since = "4.0.5", note = "please use `.into()` on a tuple instead")]
    pub fn new_named_sexp<T>(name: &str, value: &T) -> Sexp
    where
        T: IntoSexp,
    {
        let mut v = vec![];
        v.push(name.into());
        v.push(value.into_sexp());
        Sexp::List(v)
    }

    /// if the expression is a list, extract the `Vec<Sexp>`
    /// from it and swap it with Empty
    pub fn take_list(&mut self) -> Result<Vec<Sexp>, SexpError> {
        let mut e = Sexp::Empty;
        mem::swap(&mut e, self);
        match e {
            Sexp::List(v) => Ok(v),
            _ => Err(format!("Not a list: {}", e).into()),
        }
    }

    /// if the expression is a `String`, take it out and swap it with Empty
    pub fn take_string(&mut self) -> Result<String, SexpError> {
        let mut e = Sexp::Empty;
        mem::swap(&mut e, self);
        match e {
            Sexp::String(s) => Ok(s),
            _ => Err(format!("Not a string: {}", e).into()),
        }
    }

    /// create a symbolic-expression via the `IntoSexp` trait
    pub fn from<T: IntoSexp>(t: &T) -> Sexp {
        t.into_sexp()
    }

    /// access the symbolic-expression as if it is a List
    pub fn list(&self) -> Result<&Vec<Sexp>, SexpError> {
        match *self {
            Sexp::List(ref v) => Ok(v),
            _ => Err(format!("not a list: {}", self).into()),
        }
    }

    /// access the symbolic-expression as if it is an `&String`
    pub fn string(&self) -> Result<&String, SexpError> {
        match *self {
            Sexp::String(ref s) => Ok(s),
            _ => Err(format!("not a string: {}", self).into()),
        }
    }

    /// access the symbolic-expression as if it is a `String`
    pub fn s(&self) -> Result<String, SexpError> {
        match *self {
            Sexp::String(ref s) => Ok(s.clone()),
            _ => Err(format!("not a string: {}", self).into()),
        }
    }

    /// is this expression a string
    pub fn is_string(&self) -> bool {
        match *self {
            Sexp::String(_) => true,
            _ => false,
        }
    }

    /// is this expression a list
    pub fn is_list(&self) -> bool {
        match *self {
            Sexp::List(_) => true,
            _ => false,
        }
    }

    /// access the symbolic-expression as if it is a String
    /// that is a f64
    pub fn f(&self) -> Result<f64, SexpError> {
        let s = self.string()?;
        let f = s.parse()?;
        Ok(f)
    }

    /// access the symbolic-expression as if it is a String
    /// that is an i64
    pub fn i(&self) -> Result<i64, SexpError> {
        let s = self.string()?;
        let i = s.parse()?;
        Ok(i)
    }

    /// access the symbolic-expression as if it is a List
    /// assuming the first element is a String and return that
    pub fn list_name(&self) -> Result<&String, SexpError> {
        let l = self.list()?;
        let l = &l[..];
        let a = l[0].string()?;
        Ok(a)
    }

    /// access the symbolic-expression as if it is a named List
    /// where the name is provided and returns the remaining elements
    /// after the name as a slice
    #[deprecated(since = "4.1.4", note = "please use `iteratom::IterAtom::new` instead")]
    pub fn slice_atom(&self, s: &str) -> Result<&[Sexp], SexpError> {
        let v = self.list()?;
        let v2 = &v[..];
        let st = v2[0].string()?;
        if st != s {
            return Err(format!("list {} doesn't start with {}, but with {}", self, s, st).into());
        };
        Ok(&v[1..])
    }

    fn slice_atom_int(&self, s: &str) -> Result<&[Sexp], SexpError> {
        let v = self.list()?;
        let v2 = &v[..];
        let st = v2[0].string()?;
        if st != s {
            return Err(format!("list {} doesn't start with {}, but with {}", self, s, st).into());
        };
        Ok(&v[1..])
    }

    /// access the symbolic-expression as if it is a named List
    /// with two elements where the name is provided and returns
    /// the next element in the list
    pub fn named_value(&self, s: &str) -> Result<&Sexp, SexpError> {
        let v = self.list()?;
        if v.len() != 2 {
            return Err(format!("list {} is not a named_value", s).into());
        }
        let l = self.slice_atom_int(s)?;
        Ok(&l[0])
    }

    /// as named_value but converted to i64
    pub fn named_value_i(&self, s: &str) -> Result<i64, SexpError> {
        self.named_value(s)?.i()
    }

    /// as named_value but converted to f64
    pub fn named_value_f(&self, s: &str) -> Result<f64, SexpError> {
        self.named_value(s)?.f()
    }

    /// as named_value but converted to `&String`
    pub fn named_value_string(&self, s: &str) -> Result<&String, SexpError> {
        self.named_value(s)?.string()
    }

    /// as named_value but converted to `String`
    pub fn named_value_s(&self, s: &str) -> Result<String, SexpError> {
        Ok(self.named_value(s)?.string()?.clone())
    }

    /// get the symbolic-expression as a list which starts
    /// with a string that indicates the name and has num more
    /// elements, returns those elements
    #[deprecated(since = "4.1.4", note = "please use `iteratom::IterAtom::new` instead")]
    pub fn slice_atom_num(&self, s: &str, num: usize) -> Result<&[Sexp], SexpError> {
        let v = self.list()?;
        let v2 = &v[..];
        let st = v2[0].string()?;
        if st != s {
            return Err(format!("list doesn't start with {}, but with {}", s, st).into());
        };
        if v.len() != (num + 1) {
            return Err(
                format!(
                    "list ({}) doesn't have {} elements but {}",
                    s,
                    num,
                    v.len() - 1
                ).into(),
            );
        }
        Ok(&v[1..])
    }
}

impl fmt::Display for Sexp {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Sexp::String(ref s) => write!(f, "{}", encode_string(s)),
            Sexp::List(ref v) => {
                write!(f, "(")?;
                let l = v.len();
                for (i, x) in v.iter().enumerate() {
                    if i < l - 1 {
                        write!(f, "{} ", x)?;
                    } else {
                        write!(f, "{}", x)?;
                    }
                }
                write!(f, ")")
            }
            Sexp::Empty => Ok(()),
        }
    }
}

impl Default for Sexp {
    fn default() -> Sexp {
        Sexp::Empty
    }
}
