// (c) 2016-2017 Productize SPRL <joost@productize.be>

use error::SexpError;
use Sexp;
use parse_error;
use std::io;
use std::fs::File;
use std::io::prelude::*;

#[derive(Default)]
struct Parser {
    data: Vec<char>,
    position: usize,
    line: usize,
    line_position: usize,
}

impl Parser {
    fn peek(&self) -> Result<char, SexpError> {
        self.fail_on_eof()?;
        Ok(self.data[self.position])
    }

    fn get(&mut self) -> Result<char, SexpError> {
        self.fail_on_eof()?;
        let c = self.data[self.position];
        self.position += 1;
        self.line_position += 1;
        if c == '\n' {
            self.line += 1;
            self.line_position = 0;
        }
        Ok(c)
    }

    fn inc(&mut self) {
        let c = self.data[self.position];
        self.position += 1;
        self.line_position += 1;
        if c == '\n' {
            self.line += 1;
            self.line_position = 0;
        }
    }

    fn eat_space(&mut self) {
        while !self.eof() {
            let c = self.data[self.position];
            if c == ' ' || c == '\t' {
                self.inc();
                continue;
            }
            break;
        }
    }

    fn eat_char(&mut self, c: char) -> Result<(), SexpError> {
        let c2 = self.get()?;
        if c != c2 {
            self.parse_error(&format!("expected {} got {}", c, c2))
        } else {
            Ok(())
        }
    }

    fn eof(&self) -> bool {
        self.position >= self.data.len()
    }

    fn fail_on_eof(&self) -> Result<(), SexpError> {
        if self.eof() {
            return self.parse_error("End of file reached");
        }
        Ok(())
    }

    fn parse_error<T>(&self, msg: &str) -> Result<T, SexpError> {
        parse_error(self.line + 1, self.line_position + 1, msg.to_string())
    }
}

/// parse a &str to a symbolic-expression
pub fn parse_str(sexp: &str) -> Result<Sexp, SexpError> {
    if sexp.is_empty() {
        return Ok(Sexp::default());
    }
    let mut parser = Parser::default();
    parser.data = sexp.chars().collect();
    parse(&mut parser)
}

fn parse(parser: &mut Parser) -> Result<Sexp, SexpError> {
    parser.eat_space();
    let c = parser.peek()?;
    if c == '(' {
        parse_list(parser)
    } else if c == '"' {
        parse_quoted_string(parser)
    } else if c == ')' {
        parser.parse_error("Unexpected )")
    } else {
        parse_bare_string(parser)
    }
}

fn parse_list(parser: &mut Parser) -> Result<Sexp, SexpError> {
    parser.eat_char('(')?;
    let mut v = vec![];
    while !parser.eof() {
        let c = parser.peek()?;
        if c == ')' {
            break;
        } else if c == ' ' || c == '\t' || c == '\r' || c == '\n' {
            parser.inc()
        } else {
            let s = parse(parser)?;
            v.push(s)
        }
    }
    parser.eat_char(')')?;
    parser.eat_space();
    Ok(Sexp::List(v))
}

fn parse_quoted_string(parser: &mut Parser) -> Result<Sexp, SexpError> {
    let mut s = String::new();
    parser.eat_char('"')?;
    // note that escaped quotes are actually not allowed
    let mut escape = false;
    while !parser.eof() {
        let c = parser.peek()?;
        if c == '\\' {
            escape = true;
        } else if c == '"' {
            if !escape {
                break;
            } else {
                escape = false;
            }
        } else {
            escape = false;
        }
        s.push(c);
        parser.inc()
    }
    parser.eat_char('"')?;
    Ok(Sexp::String(s))
}

fn parse_bare_string(parser: &mut Parser) -> Result<Sexp, SexpError> {
    let mut s = String::new();
    while !parser.eof() {
        let c = parser.peek()?;
        if c == ' ' || c == '(' || c == ')' || c == '\r' || c == '\n' {
            break;
        }
        s.push(c);
        parser.inc()
    }
    Ok(Sexp::String(s))
}

fn read_file(name: &str) -> Result<String, io::Error> {
    let mut f = File::open(name)?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    Ok(s)
}

/// parse a file as a symbolic-expression
pub fn parse_file(name: &str) -> Result<Sexp, SexpError> {
    let s = read_file(name)?;
    parse_str(&s[..])
}
