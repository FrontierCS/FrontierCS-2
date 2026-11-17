use cstr_enum::*;

#[derive(FromCStr)]
enum Enum0 {
  A,
  B,
  C
}

#[derive(AsCStr)]
enum Enum1 {
  A,
  B,
  C
}
#[derive(FromCStr, AsCStr)]
enum Enum2 {
  A,
  B,
  C
}

#[derive(FromCStr, AsCStr)]
enum Enum3{
  #[cstr(name="apple")]
  A,
  B,
  C
}

fn main() {

}