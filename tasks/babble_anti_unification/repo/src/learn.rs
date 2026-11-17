//! The primary interface for learned library candidates.
use crate::{
    ast_node::{Arity, AstNode, PartialExpr},
    co_occurrence::CoOccurrences,
    teachable::{BindingExpr, Teachable},
};
use egg::{Analysis, EGraph, Id, Language, Pattern, Rewrite, Searcher, Var};
use log::debug;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{Debug, Display},
    num::ParseIntError,
    str::FromStr,
};
use thiserror::Error;

/// A library function's name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LibId(pub usize);

impl Display for LibId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "l{}", self.0)
    }
}

/// An error when parsing a LibId.
#[derive(Clone, Debug, Error)]
pub enum ParseLibIdError {
    /// The string did not start with "$"
    #[error("expected de Bruijn index to start with 'l")]
    NoLeadingL,
    /// The index is not a valid unsigned integer
    #[error(transparent)]
    InvalidIndex(ParseIntError),
}

impl FromStr for LibId {
    type Err = ParseLibIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(n) = s.strip_prefix('l') {
            let n = n.parse().map_err(ParseLibIdError::InvalidIndex)?;
            Ok(LibId(n))
        } else {
            Err(ParseLibIdError::NoLeadingL)
        }
    }
}

/// Signature of a pattern match in an e-graph.
/// Used to deduplicate equivalent patterns.
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Match {
    /// The e-class that the match is found in.
    class: Id,
    /// The range of the match's substitution
    /// (a multiset of class ids, stores as a sorted vector).
    actuals: Vec<Id>,
}

impl Match {
    fn new(class: Id, mut actuals: Vec<Id>) -> Self {
        actuals.sort();
        Self { class, actuals }
    }
}

/// A collection of reusable function candidates learned from an e-graph.
#[derive(Debug, Clone)]
pub struct LearnedLibrary<Op, T> {
    /// The candidate patterns discovered during construction.
    nontrivial_aus: BTreeSet<PartialExpr<Op, Var>>,
    marker: std::marker::PhantomData<T>,
}

impl<'a, Op> LearnedLibrary<Op, (Id, Id)>
where
    Op: Arity + Clone + Debug + Ord + Sync + Send + Display + 'static,
    AstNode<Op>: Language,
{
    /// Constructs reusable library candidates from a saturated e-graph.
    pub fn new<A: Analysis<AstNode<Op>>>(
        egraph: &'a EGraph<AstNode<Op>, A>,
        learn_constants: bool,
        max_arity: Option<usize>,
        co_occurrences: CoOccurrences,
    ) -> Self {
        let build_candidates = || {
            // FRONTIER_EDIT_BEGIN: candidate generation
            let _ = (egraph, learn_constants, max_arity, co_occurrences);
            todo!("library candidate generation is not implemented")
            // FRONTIER_EDIT_END: candidate generation
        };
        let library = build_candidates();
        library.assert_contract(learn_constants, max_arity);
        library
    }

    /// Enforces constructor invariants independently of candidate generation.
    /// The editable implementation runs inside a closure so an early `return`
    /// cannot bypass this check.
    fn assert_contract(&self, learn_constants: bool, max_arity: Option<usize>) {
        for candidate in &self.nontrivial_aus {
            assert!(
                learn_constants || candidate.has_holes(),
                "candidate constructor emitted a constant while constants are disabled"
            );
            assert!(
                max_arity.map_or(true, |limit| candidate.unique_holes().len() <= limit),
                "candidate constructor exceeded maximum arity"
            );
        }
    }
}

impl<Op, T> LearnedLibrary<Op, T>
where
    Op: Arity + Clone + Debug + Display + Ord + Send + Sync + Teachable + 'static,
    AstNode<Op>: Language,
{
    /// Returns an iterator over rewrite rules that replace expressions with
    /// equivalent calls to a learned library function.
    ///
    /// For example, the expression
    ///
    /// ```text
    /// (* (+ 1 2) 5)
    /// ```
    ///
    /// might be rewritten to
    ///
    /// ```text
    /// (lib f (lambda (* (+ 1 $0) 5))
    ///  (apply f 2))
    /// ```
    ///
    /// by a rewrite rule
    ///
    /// ```text
    /// (* (+ 1 ?x) 5) => (lib f (lambda (* (+ 1 $0) 5)) (apply f ?x))
    /// ```
    pub fn rewrites<A: Analysis<AstNode<Op>>>(
        &self,
    ) -> impl Iterator<Item = Rewrite<AstNode<Op>, A>> + '_ {
        self.nontrivial_aus.iter().enumerate().map(|(i, au)| {
            let searcher: Pattern<_> = au.clone().into();
            let applier: Pattern<_> = reify(LibId(i), au.clone()).into();
            let name = format!("library-candidate {}", i);
            debug!("Found rewrite \"{}\":\n{} => {}", name, searcher, applier);

            // Both patterns contain the same variables, so this can never fail.
            Rewrite::new(name, searcher, applier).unwrap_or_else(|_| unreachable!())
        })
    }

    /// Right-hand sides of library rewrites.
    pub fn libs(&self) -> impl Iterator<Item = Pattern<AstNode<Op>>> + '_ {
        self.nontrivial_aus.iter().enumerate().map(|(i, au)| {
            let applier: Pattern<_> = reify(LibId(i), au.clone()).into();
            applier
        })
    }

    /// Number of patterns learned.
    pub fn size(&self) -> usize {
        self.nontrivial_aus.len()
    }

    /// If two candidate patterns (stored in `nontrivial_aus`) have the same set of matches,
    /// only preserve the smaller one of them.
    /// Here a match is a pair of the e-class where the match was found
    /// and the range of its substitution
    /// (as a multiset of e-classes; which variables matched which e-classes is irrelevant).
    /// The reason two such patterns are equivalent is because their corresponding library functions
    /// can be used in exactly the same places and will have the same multiset (and hence size) of actual arguments.
    ///
    /// For example, after running a DSR (+ ?x ?y) => (+ ?y ?x),
    /// for any learned pattern containing (+ ?x0 ?x1), there will be an equivalent pattern containing (+ ?x1 ?x0),
    /// which will be eliminated here.
    pub fn deduplicate<A: Analysis<AstNode<Op>>>(&mut self, egraph: &EGraph<AstNode<Op>, A>) {
        // The algorithm is simply to iterate over all patterns,
        // and save their matches in a dictionary indexed by the match set.
        let mut cache: BTreeMap<Vec<Match>, PartialExpr<Op, Var>> = BTreeMap::new();
        for au in &self.nontrivial_aus {
            let pattern: Pattern<_> = au.clone().into();
            // A key in `cache` is a set of matches
            // represented as a sorted vector.
            let mut key = vec![];
            for m in pattern.search(egraph) {
                for sub in m.substs {
                    let actuals: Vec<_> = pattern.vars().iter().map(|v| sub[*v]).collect();
                    let match_signature = Match::new(m.eclass, actuals);
                    key.push(match_signature);
                }
            }
            key.sort();
            match cache.get(&key) {
                Some(cached) if cached.size() <= au.size() => {
                    debug!(
                        "Pruning pattern {}\n as a duplicate of {}",
                        pattern,
                        Pattern::from(cached.clone())
                    );
                }
                _ => {
                    cache.insert(key, au.clone());
                }
            }
        }
        self.nontrivial_aus = cache.values().cloned().collect();
    }
}

/// Converts a candidate into a partial expression which defines a new
/// named function and applies it to the candidate's metavariables.
/// The new function reifies the candidate, replacing metavariables by
/// lambda arguments.
///
/// For example, the candidate
///
/// ```text
/// (* ?x (+ ?y 1))
/// ```
///
/// would be converted to the partial expression
///
/// ```text
/// (lib l_i (lambda (lambda (* $0 (+ $1 1))))
///  (apply (apply l_i ?y) ?x))
/// ```
///
/// assuming `name` is "foo".
#[must_use]
fn reify<Op, T>(ix: LibId, au: PartialExpr<Op, T>) -> PartialExpr<Op, T>
where
    Op: Arity + Teachable,
    T: Eq,
{
    let mut metavars = Vec::new();

    // Replace every metavariable in this candidate with a de
    // Bruijn-indexed variable.
    // Metavariables might be located inside lambdas. To deal with this,
    // the de Brujin index that we return is equal to the index of the
    // metavar, added to however many lambdas wrap the metavar at that
    // point.
    let mut fun = au.fill_with_binders(|metavar, num_binders| {
        let index = metavars
            .iter()
            .position(|other: &(T, usize)| other.0 == metavar)
            .unwrap_or_else(|| {
                metavars.push((metavar, num_binders));
                metavars.len() - 1
            });
        let index = index + num_binders;

        let mut res = PartialExpr::Hole(index);

        for i in (0..num_binders).rev() {
            res = Op::apply(res, Op::var(i).into()).into();
        }

        res
    });

    // foo (\. \. $0 $2 ?hole) => foo (\. \. $0 $2 ?$2)
    //                                          ^ binders = 2

    // All the function variables
    let offset = metavars.len();

    let mut max_locals = 0;

    fun = fun.map_leaves_with_binders(|node, binders| match node.as_binding_expr() {
        Some(BindingExpr::Var(index)) if index.0 >= binders => {
            max_locals = std::cmp::max(max_locals, index.0 - binders + 1);
            Op::var(index.0 + offset).into()
        }
        _ => node.into(),
    });

    // foo (\. \. $0 $2 ?$2) => foo (\. \. $0 $3 ?$2)

    let mut fun = fun.fill(|index| Op::var(index).into());

    // Wrap that in a lambda-abstraction, one for each variable we introduced.
    for _ in 0..(metavars.len() + max_locals) {
        fun = Op::lambda(fun).into();
    }

    // Now apply the new function to the metavariables in reverse order so they
    // match the correct de Bruijn indexed variable.
    let mut body = Op::lib_var(ix).into();
    while let Some((metavar, binders)) = metavars.pop() {
        let mut fn_arg = PartialExpr::Hole(metavar);
        for _i in 0..binders {
            fn_arg = Op::lambda(fn_arg).into();
        }
        body = Op::apply(body, fn_arg).into();
    }

    for index in 0..max_locals {
        body = Op::apply(body, Op::var(index).into()).into();
    }

    PartialExpr::Node(BindingExpr::Lib(ix, fun, body).into())
}
