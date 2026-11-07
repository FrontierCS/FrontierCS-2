#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
/*!

This crate provides an easy-to-use [`SymbolTable`]
 that's fast, suitable for concurrent access,
 and provides stable `&str` references for resolved symbols.

The library supports `no_std` builds and uses the standard library by default.

With the `global` feature enabled, the
 provided [`GlobalSymbol`] type
 provides a lot of convenience methods and trait implementations
 for converting to/from strings.
*/

#[cfg(feature = "global")]
mod global;
#[cfg(feature = "global")]
pub use global::GlobalSymbol;

extern crate alloc;

use alloc::{boxed::Box, vec::Vec};
use core::{
    hash::{BuildHasher, Hash},
    num::NonZeroU32,
};

use crossbeam_utils::CachePadded;
use hashbrown::hash_map::{HashMap, RawEntryMut};
#[cfg(not(feature = "std"))]
use spin::{Mutex, MutexGuard};
#[cfg(feature = "std")]
use std::sync::{Mutex, MutexGuard};

#[inline(always)]
fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    #[cfg(feature = "std")]
    {
        mutex.lock().unwrap()
    }
    #[cfg(not(feature = "std"))]
    {
        mutex.lock()
    }
}

/// A `BuildHasher` that builds a determinstically seeded hasher.
#[derive(Default)]
pub struct DeterministicHashBuilder;

impl BuildHasher for DeterministicHashBuilder {
    type Hasher = foldhash::fast::FoldHasher;
    fn build_hasher(&self) -> Self::Hasher {
        foldhash::fast::FixedState::with_seed(0).build_hasher()
    }
}

/// The default number of sharded in the [`SymbolTable`].
pub const DEFAULT_N_SHARDS: usize = 16;

/// A table in which you can intern strings and get back [`Symbol`]s.
///
/// The table is sharded `N` times (default [`DEFAULT_N_SHARDS`])
/// for lower contention when accessing concurrently.
pub struct SymbolTable<const N: usize = DEFAULT_N_SHARDS, S = DeterministicHashBuilder> {
    build_hasher: S,
    shards: [CachePadded<Mutex<Shard>>; N],
}

impl<const N: usize, S> SymbolTable<N, S> {
    const SHARD_BITS: u32 = 32 - (N as u32 - 1).leading_zeros();
    const MAX_IDX: u32 = u32::MAX >> Self::SHARD_BITS;
}

impl SymbolTable<DEFAULT_N_SHARDS, DeterministicHashBuilder> {
    /// Creates a new [`SymbolTable`] with the default generic arguments.
    ///
    /// This symbol table will be determinisitic, using a seeded hasher.
    ///
    /// To create a [`SymbolTable`] with a custom shard count or hasher, use
    /// [`with_hasher`](Self::with_hasher).
    pub const fn new() -> Self {
        Self::with_hasher(DeterministicHashBuilder)
    }
}

impl<const N: usize, S: BuildHasher> SymbolTable<N, S> {
    /// Creates a new [`SymbolTable`] with a custom hasher.
    #[allow(clippy::assertions_on_constants)]
    pub const fn with_hasher(build_hasher: S) -> Self {
        // Only used for array initialization.
        // Replace with an inline-const block when it's been stable for long enough.
        #[allow(clippy::declare_interior_mutable_const)]
        const SHARD_INIT: CachePadded<Mutex<Shard>> = CachePadded::new(Mutex::new(Shard::new()));

        assert!(0 < N);
        assert!(N <= 1024);
        // println!("N = {}", N);
        // println!("SHARD_BITS = {}", Self::SHARD_BITS);
        // println!("MAX_IDX = {}", Self::MAX_IDX);
        Self {
            build_hasher,
            shards: [SHARD_INIT; N],
        }
    }
}

#[derive(Default)]
struct Shard {
    map: HashMap<u32, (), ()>,
    strs: Vec<Box<str>>,
}

impl Shard {
    const fn new() -> Self {
        Self {
            map: HashMap::with_hasher(()),
            strs: Vec::new(),
        }
    }

    fn intern(&mut self, hash: u64, string: &str, build_hasher: &impl BuildHasher) -> u32 {
        let entry = self
            .map
            .raw_entry_mut()
            .from_hash(hash, |&idx| string == self.strs[idx as usize].as_ref());

        let index = match entry {
            RawEntryMut::Occupied(e) => *e.key(),
            RawEntryMut::Vacant(e) => {
                let idx = self.strs.len() as u32;
                self.strs.push(string.into());

                *e.insert_with_hasher(hash, idx, (), |&idx| {
                    hash_one(build_hasher, self.strs[idx as usize].as_ref())
                })
                .0
            }
        };

        debug_assert!(!self.strs.is_empty());
        debug_assert!(!self.map.is_empty());
        index
    }

    fn get_existing(&self, hash: u64, string: &str) -> Option<u32> {
        let entry = self
            .map
            .raw_entry()
            .from_hash(hash, |&idx| string == self.strs[idx as usize].as_ref());

        entry.map(|(e, _)| *e)
    }
}

impl<const N: usize, S: Default + BuildHasher> Default for SymbolTable<N, S> {
    fn default() -> Self {
        Self::with_hasher(S::default())
    }
}

#[inline(always)]
fn hash_one(build_hasher: &impl BuildHasher, string: &str) -> u64 {
    let mut hasher = build_hasher.build_hasher();
    string.hash(&mut hasher);
    core::hash::Hasher::finish(&hasher)
}

impl<const N: usize, S: BuildHasher> SymbolTable<N, S> {
    /// Intern a string into the [`SymbolTable`].
    ///
    /// Note how this method only takes `&self`, so it can be used concurrently.
    ///
    /// Interning the same string will give the same symbol.
    ///
    /// ```
    /// let mut table = symbol_table::SymbolTable::new();
    /// assert_eq!(table.intern("foo"), table.intern("foo"));
    /// ```
    pub fn intern(&self, string: &str) -> Symbol {
        let hash = hash_one(&self.build_hasher, string);
        let shard_i = hash as usize % N;
        // println!("Interning into shard {shard_i}");

        let mut locked = lock_mutex(&self.shards[shard_i]);
        let i = locked.intern(hash, string, &self.build_hasher) + 1;
        drop(locked);

        assert!(i < Self::MAX_IDX, "Can't represent index {} in a Symbol", i);
        let shard_bits: u32 = (shard_i as u32) << (32 - Self::SHARD_BITS);
        // println!("shard_bits = {shard_bits:x}");
        Symbol(NonZeroU32::new(shard_bits | i).unwrap())
    }

    /// Get the [`Symbol`] for a string in the [`SymbolTable`], only if it already exists.
    ///
    /// Note how this method only takes `&self`, so it can be used concurrently.
    ///
    /// Interning the same string will give the same symbol.
    ///
    /// ```
    /// let mut table = symbol_table::SymbolTable::new();
    /// assert!(table.get_existing("foo").is_none());
    /// table.intern("foo");
    /// assert!(table.get_existing("foo").is_some());
    /// ```
    pub fn get_existing(&self, string: &str) -> Option<Symbol> {
        let hash = hash_one(&self.build_hasher, string);
        let shard_i = hash as usize % N;
        // println!("Interning into shard {shard_i}");

        let locked = lock_mutex(&self.shards[shard_i]);
        let i = locked.get_existing(hash, string)? + 1;
        drop(locked);

        assert!(i < Self::MAX_IDX, "Can't represent index {} in a Symbol", i);
        let shard_bits: u32 = (shard_i as u32) << (32 - Self::SHARD_BITS);
        // println!("shard_bits = {shard_bits:x}");
        Some(Symbol(NonZeroU32::new(shard_bits | i).unwrap()))
    }

    /// Resolve a symbol to the interned string.
    ///
    /// The resolved string is immutable and will live as long as the
    /// [`SymbolTable`].
    ///
    /// ```
    /// let mut table = symbol_table::SymbolTable::new();
    /// let foo = table.intern("foo");
    /// assert_eq!(table.resolve(foo), "foo");
    /// ```
    pub fn resolve(&self, sym: Symbol) -> &str {
        let shard_i = sym.0.get() >> (32 - Self::SHARD_BITS);
        debug_assert!(shard_i < N as u32);
        // println!("Resolving from shard {shard_i}");
        let i = sym.0.get() & (u32::MAX >> Self::SHARD_BITS);
        debug_assert!(i > 0);
        let i = i - 1; // undo the + 1 from interning
        let shard = lock_mutex(&self.shards[shard_i as usize]);
        debug_assert!(
            !shard.strs.is_empty(),
            "Shard shouldn't be empty when resolving!"
        );
        let str: &str = &shard.strs[i as usize];

        // SAFETY:
        // We can "extend" the lifetime of str outside the mutex lock
        // because we know it will never move or be mutated. The only thing to
        // worry about is it getting dropped, but that's ok because it's
        // lifetime is less than `self`.
        unsafe { &*(str as *const str) }
    }
}

/// An interned symbol.
///
/// Resolve it back to the string by using [`SymbolTable::resolve`]
///
/// Internally, this is a [`NonZeroU32`], so it will be niche-optimized.
///
/// ```
/// # use core::mem::size_of; use symbol_table::Symbol;
/// assert_eq!(size_of::<Symbol>(), size_of::<u32>());
/// ```
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol(NonZeroU32);

impl From<NonZeroU32> for Symbol {
    fn from(i: NonZeroU32) -> Self {
        Symbol(i)
    }
}

impl From<Symbol> for NonZeroU32 {
    fn from(sym: Symbol) -> Self {
        sym.0
    }
}
