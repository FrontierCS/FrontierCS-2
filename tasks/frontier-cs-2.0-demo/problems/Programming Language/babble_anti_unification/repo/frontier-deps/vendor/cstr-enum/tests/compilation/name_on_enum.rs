use cstr_enum::*;

#[derive(FromCStr, AsCStr)]
#[cstr(name="egg")]
enum Enum {
  A
}

fn main() {

}