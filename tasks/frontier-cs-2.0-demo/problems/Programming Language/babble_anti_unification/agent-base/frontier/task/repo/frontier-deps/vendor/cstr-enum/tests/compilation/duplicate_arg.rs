use cstr_enum::*;

#[derive(AsCStr)]
enum Enum0 {
  #[cstr(name="egg")]
  #[cstr(name="egg")]
  A,
  B,
  C
}

#[derive(AsCStr)]
enum Enum1 {
  #[cstr(name="egg", name="foo")]
  A,
  B,
  C
}

fn main() {

}