use cstr_enum::*;

#[derive(AsCStr)]
enum Enum0 {
  A(u8),
  B,
  C
}

#[derive(AsCStr)]
enum Enum1 {
  A,
  B{ foo: u8, bar: u16},
  C
}

fn main() {

}