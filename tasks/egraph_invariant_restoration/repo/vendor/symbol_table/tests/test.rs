use std::collections::{hash_map::Entry, HashMap};
use symbol_table::*;

static TEXT: &str = include_str!("../tests/gulliver.txt");

#[test]
fn test_resolve() {
    let interner = SymbolTable::new();
    let mut resolved = Vec::new();
    let mut map = HashMap::new();

    for word in TEXT.split_whitespace().chain(TEXT.split_whitespace().rev()) {
        let sym = interner.intern(word);
        match map.entry(word) {
            Entry::Occupied(e) => assert_eq!(*e.get(), sym),
            Entry::Vacant(e) => {
                e.insert(sym);
            }
        }
        resolved.push(interner.resolve(sym));
        assert_eq!(interner.resolve(sym), word);
    }
}

#[cfg(feature = "global")]
#[test]
fn test_global() {
    let mut map = HashMap::new();

    for word in TEXT.split_whitespace().chain(TEXT.split_whitespace().rev()) {
        let sym = GlobalSymbol::from(word);
        match map.entry(word) {
            Entry::Occupied(e) => {
                assert_eq!(*e.get(), sym);
            }
            Entry::Vacant(e) => {
                e.insert(sym);
            }
        }

        assert_eq!(sym.to_string(), word);
        assert_eq!(sym.as_str(), word);
    }
}

#[test]
fn test_specific_strings() {
    let interner = SymbolTable::new();
    let strings = ["", "asdf", "🧵"];
    for word in strings {
        let sym = interner.intern(word);
        assert_eq!(interner.resolve(sym), word);
    }
}

#[cfg(feature = "global")]
#[cfg(feature = "serde")]
#[test]
fn test_serde_serialization_deserialization() {
    let sym = GlobalSymbol::from("foo");

    fn ser(_: impl serde::Serialize) {}
    fn de<'a>(_: impl serde::Deserialize<'a>) {}

    ser(sym);
    de(sym);
}

#[cfg(feature = "global")]
#[cfg(feature = "serde")]
#[test]
fn test_serde_file() {
    let test = GlobalSymbol::from("foo");

    let mut file: std::fs::File = tempfile::tempfile().expect("Failed to create tempfile");

    // serialize symbol to file
    serde_json::to_writer(&mut file, &test).expect("Failed to serialize");

    // seek back to the beginning of the file
    std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(0)).expect("Failed to seek");

    // deserialize the symbol back out from the file
    let deserialized = serde_json::from_reader(file).expect("Failed to deserialize");

    assert_eq!(test, deserialized);
}

#[cfg(feature = "global")]
#[cfg(feature = "serde")]
#[test]
fn test_serde_string() {
    let test = symbol_table::static_symbol!("foo");

    let ser = serde_json::to_string(&test).expect("Failed to serialize");
    let de = serde_json::from_str(&ser).expect("Failed to deserialize");

    assert_eq!(test, de);
}
