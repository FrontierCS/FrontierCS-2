#![warn(missing_docs)]
//! A crate for defining C-style string enums.
//!
//! C APIs sometimes require string constants.  One could define a bunch of `&CStr` constants using the
//! [`constr_cstr`](https://docs.rs/const-cstr/) crate, but this becomes unergonomic with a large number of constants.
//! It also does not allow the type checking Rust's enums provide.
//!
//! This crate provides two traits for converting between to and from `&CStr`: `AsCStr` and `FromCStr`.  It also provides
//! derive macros for implementing these traits on enums.  The implementations provided
//! by the derive macros perform no allocations, using only static `[u8]` buffers.
//!
//! ```
//! use cstr_enum::*;
//! use std::ffi::CStr;
//! use std::os::raw::c_char;
//!
//! #[derive(Debug, Eq, PartialEq, FromCStr, AsCStr)]
//! enum Constants {
//!   Apple,
//!   Bacon,
//!   Cat = 1337, // user discriminants supported
//! }
//!
//! assert_eq!(Constants::Apple.as_cstr().to_bytes_with_nul(), b"Apple\0");
//!
//! let returned_from_c_api = CStr::from_bytes_with_nul(b"Cat\0").unwrap();
//! assert_eq!(Constants::from_cstr(returned_from_c_api), Ok(Constants::Cat));
//!
//! let returned_from_c_api = CStr::from_bytes_with_nul(b"unknown\0").unwrap();
//! assert_eq!(
//!   Constants::from_cstr(returned_from_c_api),
//!   Err("unexpected string while parsing for Constants variant")
//! );
//! ```
//! Both derive macros allow the re-naming of enum variants using the `cstr(name="string literal")` attribute on enum variants.
//! ```
//! # use cstr_enum::*;
//! #
//! #[derive(Debug, Eq, PartialEq, FromCStr, AsCStr)]
//! enum Constants {
//!   #[cstr(name="pork")]
//!   Bacon,
//! }
//!
//! assert_eq!(Constants::Bacon.as_cstr().to_bytes_with_nul(), b"pork\0");
//! ```
//! Nul bytes in the supplied string will be rejected at compile time.
//! ```compile_fail
//! # use cstr_enum::*;
//! #
//! #[derive(Debug, Eq, PartialEq, FromCStr, AsCStr)]
//! enum Constants {
//!   #[cstr(name="p\0rk")]
//!   Bacon,
//! }
//! ```
//! ```text
//! error: string cannot contain nul bytes
//!   |   #[cstr(name="p\0rk")]
//!   |               ^^^^^^^
//! ```
//! When deriving `AsCStr`, enum variants may contain fields:
//! ```
//! # use cstr_enum::*;
//! #
//! #[derive(Debug, AsCStr)]
//! enum Constants {
//!   Foo{ bar: u8 },
//!   Baz(u8, u16)
//! }
//! assert_eq!(Constants::Foo{ bar: 0 }.as_cstr().to_bytes_with_nul(), b"Foo\0");
//! assert_eq!(Constants::Baz(0,0).as_cstr().to_bytes_with_nul(), b"Baz\0");
//! ```
//! This is not the case deriving `FromCStr`:
//! ```compile_fail
//! # use cstr_enum::*;
//! #
//! #[derive(Debug, FromCStr)]
//! enum Constants {
//!   Foo{ bar: u8 },
//!   Baz(u8, u16)
//! }
//! ```
//! ```text
//! error: variant cannot have fields
//!   |   Foo{ bar: u8 },
//!   |   ^^^^^^^^^^^^^^
//! ```
//!
//! Conversion between Rust strings ([`str`] and [`String`]) is not supported by this crate. Instead, check out
//! the [`strum`](https://docs.rs/strum/) crate.
use std::ffi::CStr;

/// Conversion to a C-style string.
///
/// If using the derive macro, this will be a cheap conversion.
pub trait AsCStr {
  /// Represent self as a [`&CStr`](std::ffi::CStr)
  fn as_cstr(&self) -> &CStr;
}

/// Conversion from a C-style string
///
/// This trait should be used the same way as [`std::str::FromStr`], although
/// a separate `.parse()` implementation is not provided `&str`
pub trait FromCStr {
  /// The error type returned if parsing fails.
  ///
  /// If using the derive macro, this will be `&'static str`.
  type Err : Sized;
  /// Parse the `&CStr` for an instance of `Self`.
  ///
  /// If using the derive macro, this will be a `match` statement over `&'static [u8]`.
  fn from_cstr(s: &CStr) -> Result<Self, Self::Err> where Self: Sized;
}

pub use cstr_enum_derive::*;
