# cstr-enum ![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/ykrist/cstr-enum?sort=semver)

A crate for defining C-style string enums.

C APIs sometimes require string constants.  One could define a bunch of `&CStr` constants using the
[`constr_cstr`](https://docs.rs/const-cstr/) crate, but this becomes unergonomic with a large number of constants.
It also does not allow the type checking Rust's enums provide.

This crate provides two traits for converting between to and from `&CStr`: `AsCStr` and `FromCStr`.  It also provides
derive macros for implementing these traits on enums.  The implementations provided
by the derive macros perform no allocations, using only static `[u8]` buffers.

## Example usage
```rust
use cstr_enum::*;
use std::ffi::CStr;
use std::os::raw::c_char;

#[derive(Debug, Eq, PartialEq, FromCStr, AsCStr)]
enum Constants {
  Apple,
  Bacon,
  Cat = 1337, // user discriminants supported
}

assert_eq!(Constants::Apple.as_cstr().to_bytes_with_nul(), b"Apple\0");

let returned_from_c_api = CStr::from_bytes_with_nul(b"Cat\0").unwrap();
assert_eq!(Constants::from_cstr(returned_from_c_api), Ok(Constants::Cat));

let returned_from_c_api = CStr::from_bytes_with_nul(b"unknown\0").unwrap();
assert_eq!(
  Constants::from_cstr(returned_from_c_api),
  Err("unexpected string while parsing for Constants variant")
);
```

## License
Licensed under [`MIT`](LICENSE.txt).
