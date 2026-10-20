use crate::*;

use alloc::string::String;
use core::{convert::Infallible, fmt, num::NonZeroU32, str::FromStr};

#[cfg(feature = "global")]
/// Macro for creating symbols from &'static str. Useful for commonly used symbols known at compile time.
/// This is faster then GlobalSymbol::from(s) by avoiding mutex contention.
///
/// # Examples
///
/// ```
/// use symbol_table::static_symbol;
/// use symbol_table::GlobalSymbol;
///
/// let hello = static_symbol!("hello");
/// assert_eq!(hello, GlobalSymbol::from("hello"));
///
/// // The same symbol is returned on subsequent calls
/// let hello2 = static_symbol!("hello");
/// assert_eq!(hello, hello2);
/// ```
#[macro_export]
macro_rules! static_symbol {
    ($s:literal) => {{
        #[cfg(feature = "std")]
        {
            use std::sync::OnceLock;
            static SYMBOL: OnceLock<$crate::GlobalSymbol> = OnceLock::new();

            *SYMBOL.get_or_init(|| $crate::GlobalSymbol::from($s))
        }

        #[cfg(not(feature = "std"))]
        {
            use spin::Once;
            static SYMBOL: Once<$crate::GlobalSymbol> = Once::new();

            *SYMBOL.call_once(|| $crate::GlobalSymbol::from($s))
        }
    }};
}

/// A interned string in the global symbol table.
///
/// This requires the `global` feature on the crate.
///
/// [`GlobalSymbol`] is a wrapper around [`Symbol`] that knows to refer to a
/// built-in, global [`SymbolTable`]. Strings into the global table are never freed.
///
/// This enables a lot of convenience methods and trait implementations over
/// [`GlobalSymbol`] (see below). In particular,
///   you can convert it to `&'static str`,
///   convert [`From`] and [`Into`] a `&str`,
///   and de/serialize using [`serde`](https://serde.rs) if the `serde` feature is enabled.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(into = "&'static str"))]
pub struct GlobalSymbol(Symbol);

impl From<NonZeroU32> for GlobalSymbol {
    fn from(n: NonZeroU32) -> Self {
        Self(Symbol::from(n))
    }
}

impl From<GlobalSymbol> for NonZeroU32 {
    fn from(n: GlobalSymbol) -> Self {
        n.0.into()
    }
}

static SINGLETON: SymbolTable = SymbolTable::new();

impl GlobalSymbol {
    /// Intern a string into the global symbol table.
    pub fn new(s: impl AsRef<str>) -> Self {
        s.as_ref().into()
    }

    /// Get the symbol for a string in the global symbol table, only if it already exists.
    pub fn get_existing(s: impl AsRef<str>) -> Option<Self> {
        SINGLETON.get_existing(s.as_ref()).map(GlobalSymbol)
    }

    /// Convert this symbol into the string in the static, global symbol table.
    pub fn as_str(&self) -> &'static str {
        (*self).into()
    }
}

impl From<&str> for GlobalSymbol {
    fn from(s: &str) -> Self {
        GlobalSymbol(SINGLETON.intern(s))
    }
}

impl From<String> for GlobalSymbol {
    fn from(s: String) -> Self {
        s.as_str().into()
    }
}

impl From<&String> for GlobalSymbol {
    fn from(s: &String) -> Self {
        s.as_str().into()
    }
}

impl FromStr for GlobalSymbol {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(s.into())
    }
}

impl From<GlobalSymbol> for &'static str {
    fn from(sym: GlobalSymbol) -> Self {
        SINGLETON.resolve(sym.0)
    }
}

impl fmt::Debug for GlobalSymbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

impl fmt::Display for GlobalSymbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}

#[cfg(feature = "serde")]
struct StrVisitor;

#[cfg(feature = "serde")]
impl serde::de::Visitor<'_> for StrVisitor {
    type Value = GlobalSymbol;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a &str")
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(v.into())
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for GlobalSymbol {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_str(StrVisitor)
    }
}
