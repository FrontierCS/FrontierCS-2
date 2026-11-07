use cstr_enum::*;

#[derive(AsCStr)]
enum Enum0 {
  #[cstr(name=0)]
  A,
  B,
  C
}

#[derive(AsCStr)]
enum Enum1 {
  #[cstr(foo="egg")]
  A,
  B,
  C
}

fn main() {

}