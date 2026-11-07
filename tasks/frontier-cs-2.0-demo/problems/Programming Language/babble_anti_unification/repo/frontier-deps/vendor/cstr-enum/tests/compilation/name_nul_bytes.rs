use cstr_enum::*;

#[derive(FromCStr, AsCStr)]
enum Enum{
  #[cstr(name="app\0le")]
  A,
}

fn main() {

}