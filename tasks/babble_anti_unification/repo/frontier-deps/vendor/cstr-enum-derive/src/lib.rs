use quote::{quote, ToTokens};
use proc_macro2::{Span};
use syn::parse::{Result, Error};
use std::default::Default;
use std::ffi::CStr;


#[derive(Default)]
struct VariantMeta {
  pub name: Option<syn::LitByteStr>,
}

impl VariantMeta {
  /// Build the enum variant meta info from attributes
  pub fn from_attrs(attrs: &[syn::Attribute]) -> Result<Self> {
    let mut opts = VariantMeta::default();

    for attr in attrs {
      if attr.path.is_ident("cstr") {
        opts.parse_meta(attr.parse_meta()?)?
      }
    }
    Ok(opts)
  }

  /// Parse a single #[cstr(...)] item on a variant
  pub fn parse_meta(&mut self, meta: syn::Meta) -> Result<()> {
    match meta {
      syn::Meta::List(nvs) => {
        for nv in nvs.nested {
          match nv {
            syn::NestedMeta::Meta(syn::Meta::NameValue(nv)) => self.parse_nv(nv)?,
            _ => return Err(Error::new_spanned(nv, "expected named argument (KEY = VALUE)"))
          }
        }
      }
      _ => return Err(Error::new_spanned(meta, "missing arguments: expected `cstr(...)`"))
    }
    Ok(())
  }

  /// Parse a single item in the list of name-value pairs inside the #[cstr(...)]
  fn parse_nv(&mut self, nv: syn::MetaNameValue) -> Result<()> {
    if let Some(ident) = nv.path.get_ident() {
      if ident == "name" {
        Self::check_not_set(&self.name, ident)?;
        match nv.lit {
          syn::Lit::Str(s) => {
            let mut name = s.value();
            name.push('\0');
            if CStr::from_bytes_with_nul(name.as_bytes()).is_err() {
              return Err(Error::new_spanned(s, "string cannot contain nul bytes"));
            }
            self.name = Some(syn::LitByteStr::new(name.as_bytes(), s.span()));
            return Ok(());
          }
          lit => { return Err(Error::new_spanned(lit, "expected string literal")); }
        }
      }
      // future attributes can be added here.  Annoyingly, a match statement doesn't work
      // since `ident` is of a different type
      // ...
    }
    Err(Error::new_spanned(nv.path, "invalid named argument"))
  }

  /// Check the field hasn't been set before by another attribute item
  fn check_not_set<T>(field: &Option<T>, tokens: impl ToTokens) -> Result<()> {
    if field.is_some() {
      Err(Error::new_spanned(tokens, "duplicate named argument"))
    } else {
      Ok(())
    }
  }
}

/// Convert an ident to a nul-terminated byte-string literal.
fn ident_to_byte_str_lit(ident: &syn::Ident) -> syn::LitByteStr {
  let cstring = {
    let mut s = ident.to_string();
    s.push('\0');
    s
  };
  syn::LitByteStr::new(cstring.as_bytes(), Span::call_site())
}

/// Check that #[cstr(...)] is not applied to the enum itself
fn check_enum_attrs(input: &syn::DeriveInput) -> Result<()> {
  for attr in &input.attrs {
    if attr.path.is_ident("cstr") {
      return Err(Error::new_spanned(attr, "attribute must be placed on variants"));
    }
  }
  Ok(())
}

/// Retrieve the name mapping between enum variants and their CStr representations
fn get_name_mapping<'a>(input: &'a syn::DeriveInput, unit_variants_only: bool) -> Result<(Vec<&'a syn::Ident>, Vec<syn::LitByteStr>)> {
  check_enum_attrs(input)?;

  let variants = match &input.data {
    syn::Data::Enum(enm) => &enm.variants,
    _ => return Err(Error::new(Span::call_site(), "target must be an enum")),
  };

  let mut idents = Vec::with_capacity(variants.len());
  let mut bytestrs = Vec::with_capacity(variants.len());

  #[allow(unused_variables)]
  for variant in variants {
    if unit_variants_only && variant.fields != syn::Fields::Unit {
      return Err(Error::new_spanned(variant, "variant cannot have fields"));
    }
    // parse name from attributes
    let ident = &variant.ident;
    let opts = VariantMeta::from_attrs(&variant.attrs)?;

    // Default to the ident of the variant
    bytestrs.push(opts.name.unwrap_or_else(|| ident_to_byte_str_lit(&ident)));
    idents.push(ident);
  }
  Ok((idents, bytestrs))
}


/// Derive macro for the [`AsCStr`] trait.  May only be applied to enums.
#[proc_macro_derive(AsCStr, attributes(cstr))]
pub fn derive_ascstr_enum(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
  let input = syn::parse_macro_input!(input as syn::DeriveInput);


  let (var_idents, vals) = match get_name_mapping(&input, false) {
    Ok(m) => m,
    Err(e) => { return e.to_compile_error().into(); }
  };

  let ident = &input.ident;

  let ts = quote! {
       impl cstr_enum::AsCStr for #ident {
            fn as_cstr(&self) -> &'static std::ffi::CStr {
                match self {
                    #( Self::#var_idents{..} => unsafe {std::ffi::CStr::from_bytes_with_nul_unchecked(#vals) }, )*
                }
            }
       }
    };

  ts.into()
}


/// Derive macro for the [`FromCStr`] trait.  May only be applied to enums whose variants have no fields.
#[proc_macro_derive(FromCStr, attributes(cstr))]
pub fn derive_fromcstr_enum(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
  let input = syn::parse_macro_input!(input as syn::DeriveInput);

  let (var_idents, mut vals) = match get_name_mapping(&input, true) {
    Ok(m) => m,
    Err(e) => { return e.to_compile_error().into(); }
  };

  for v in vals.iter_mut() {
    let bytes = v.value();
    *v = syn::LitByteStr::new(&bytes[..bytes.len() - 1], v.span())
  }


  let ident = &input.ident;
  let error_msg = syn::LitStr::new(&format!("unexpected string while parsing for {} variant", ident), Span::call_site());

  let ts = quote! {
       impl cstr_enum::FromCStr for #ident {
            type Err = &'static str;
            fn from_cstr(s: &std::ffi::CStr) -> Result<Self, Self::Err> {
                match s.to_bytes() {
                    #( #vals => Ok(Self::#var_idents), )*
                    _ => Err(#error_msg)
                }
            }
       }
    };

  ts.into()
}

