use std::convert::TryFrom;
use std::convert::TryInto;
use std::str::FromStr;

use crate::language::from_relay::{
    access_concatenate, access_insert_axis, access_pair, access_shape, compute,
};
use crate::language::{from_relay, ShapeData};

use super::{ComputeType, Language, MyAnalysis, MyAnalysisData, PadType, RangeSet2};
use egg::Symbol;
use egg::{
    rewrite, Applier, ConditionalApplier, EGraph, ENodeOrVar, Id, Pattern, Rewrite, SearchMatches,
    Searcher, Subst, Var,
};
use egg::{PatternAst, RecExpr};
use itertools::Itertools;
use ndarray::Dimension;
use ndarray::IxDyn;

type EG = EGraph<Language, MyAnalysis>;
type RW = Rewrite<Language, MyAnalysis>;

fn constrain_access(
    access: Var,
    constraint: impl Fn(&super::language::AccessPatternData) -> bool,
) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
    move |egraph, _, subst| match &egraph[subst[access]].data {
        MyAnalysisData::AccessPattern(a) => constraint(a),
        _ => false,
    }
}


pub fn flatten_unflatten_any_access() -> RW {
    rewrite!("flatten-unflatten-all-accesses";
             "?access" => "(access-reshape (access-flatten ?access) (get-access-shape ?access))"
             if is_access())
}

pub fn bubble_reshape_through_cartesian_product() -> RW {
    fn access_item_shapes_equal(
        left: Var,
        right: Var,
    ) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
        move |egraph, _, subst| match (&egraph[subst[left]].data, &egraph[subst[right]].data) {
            (MyAnalysisData::AccessPattern(left), MyAnalysisData::AccessPattern(right)) => {
                left.item_shape == right.item_shape
            }
            _ => false,
        }
    }

    struct ApplierImpl {
        left_shape: Var,
        right_shape: Var,
    }
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(
            &self,
            egraph: &mut EG,
            eclass: Id,
            subst: &Subst,
            _searcher_ast: Option<&PatternAst<Language>>,
            _rule_name: Symbol,
        ) -> Vec<Id> {
            let left_shape = match &egraph[subst[self.left_shape]].data {
                MyAnalysisData::AccessPattern(a) => a,
                _ => panic!(),
            };
            let right_shape = match &egraph[subst[self.right_shape]].data {
                MyAnalysisData::AccessPattern(a) => a,
                _ => panic!(),
            };

            // TODO(@gussmith23) Duplicated logic for cartprod shape calculation

            assert_eq!(
                left_shape.item_shape, right_shape.item_shape,
                "Cartesian product argument shapes must match"
            );

            let new_shape = IxDyn(
                left_shape
                    .shape
                    .as_array_view()
                    .iter()
                    .cloned()
                    .chain(right_shape.shape.as_array_view().iter().cloned())
                    .collect::<Vec<usize>>()
                    .as_slice(),
            );
            let new_item_shape = IxDyn(
                std::iter::once(2)
                    .chain(left_shape.item_shape.as_array_view().iter().cloned())
                    .collect::<Vec<usize>>()
                    .as_slice(),
            );

            assert_eq!(
                new_shape.as_array_view().iter().product::<usize>()
                    * new_item_shape.as_array_view().iter().product::<usize>(),
                left_shape.shape.as_array_view().iter().product::<usize>()
                    * right_shape.shape.as_array_view().iter().product::<usize>()
                    * 2
                    * left_shape
                        .item_shape
                        .as_array_view()
                        .iter()
                        .product::<usize>()
            );

            format!(
                "(access-reshape
                      (access-cartesian-product
                       ?left-access
                       ?right-access
                      )
                      (access-shape
                       (shape {})
                       (shape {})
                      )
                     )",
                new_shape
                    .slice()
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
                new_item_shape
                    .slice()
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            )
            .parse::<Pattern<Language>>()
            .unwrap()
            .apply_one(egraph, eclass, subst, _searcher_ast, _rule_name)
        }
    }

    rewrite!("bubble-reshape-through-cartesian-product";
             "(access-cartesian-product
               (access-reshape
                ?left-access
                ?left-shape
               )
               (access-reshape
                ?right-access
                ?right-shape
               )
              )" =>
             { ApplierImpl {
                 left_shape: "?left-shape".parse().unwrap(),
                 right_shape: "?right-shape".parse().unwrap(),
             }}
             if access_item_shapes_equal("?left-access".parse().unwrap(),
                                         "?right-access".parse().unwrap()))
}

pub fn bubble_reshape_through_compute_dot_product() -> RW {
    fn is_dot_product(op: Var) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
        move |egraph, _, subst| match &egraph[subst[op]].data {
            MyAnalysisData::ComputeType(c) => *c == super::language::ComputeType::DotProduct,
            _ => false,
        }
    }
    struct ApplierImpl(Var);
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(
            &self,
            egraph: &mut EG,
            eclass: Id,
            subst: &Subst,
            _searcher_ast: Option<&PatternAst<Language>>,
            _rule_name: Symbol,
        ) -> Vec<Id> {
            let a = match &egraph[subst[self.0]].data {
                MyAnalysisData::AccessPattern(a) => a,
                _ => panic!(),
            };

            format!(
                "(access-reshape (compute ?op ?a) (access-shape (shape {}) (shape)))",
                a.shape
                    .as_array_view()
                    .iter()
                    .map(|u: &usize| { format!("{}", u) })
                    .collect::<Vec<_>>()
                    .join(" ")
            )
            .parse::<Pattern<Language>>()
            .unwrap()
            .apply_one(egraph, eclass, subst, _searcher_ast, _rule_name)
        }
    }
    rewrite!("bubble-reshape-through-compute";
             "(compute ?op (access-reshape ?a ?shape))" =>
             { ApplierImpl("?shape".parse().unwrap()) }
             if is_dot_product("?op".parse().unwrap()))
}

pub fn bubble_access_concatenate_through_compute_dot_product_not_item_axis(
) -> Rewrite<Language, MyAnalysis> {
    rewrite!("bubble-access-concatenate-through-compute-dot-product-not-item-axis";
             "(compute dot-product (access-concatenate ?a0 ?a1 ?axis))" =>
             "(access-concatenate
               (compute dot-product ?a0)
               (compute dot-product ?a1)
               ?axis
              )"
             if not_item_axis("?axis".parse().unwrap(), "?a0".parse().unwrap()))
}

pub fn systolic_array() -> Rewrite<Language, MyAnalysis> {
    struct ApplierImpl {
        a: Var,
        b: Var,
    }
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(
            &self,
            egraph: &mut EG,
            eclass: Id,
            subst: &Subst,
            _searcher_ast: Option<&PatternAst<Language>>,
            _rule_name: Symbol,
        ) -> Vec<Id> {
            let (a, b) = match (&egraph[subst[self.a]].data, &egraph[subst[self.b]].data) {
                (MyAnalysisData::AccessPattern(a), MyAnalysisData::AccessPattern(b)) => (a, b),
                _ => panic!(),
            };
            assert_eq!(a.item_shape.ndim(), 1);
            assert_eq!(b.shape.ndim(), 1);
            assert_eq!(b.item_shape.ndim(), 1);
            let rows: usize = b.item_shape.slice()[0];
            let cols: usize = b.shape.slice()[0];

            let pattern: Pattern<Language> = format!(
                "(systolic-array {} {}
                          ?access-1
                          (access (access-transpose ?access-2 (list 1 0)) 0)
                         )",
                rows, cols
            )
            .parse()
            .unwrap();

            pattern.apply_one(egraph, eclass, subst, _searcher_ast, _rule_name)
        }
    }

    rewrite!("systolic-array";
             "(compute dot-product
               (access-cartesian-product
                ?access-1
                ?access-2
               )
              )
             " =>
             { ApplierImpl{a: "?access-1".parse().unwrap(), b: "?access-2".parse().unwrap(),}}
             if constrain_access("?access-1".parse().unwrap(),
                                 |a| a.shape.ndim() <= 1 && a.item_shape.ndim() == 1)
             if constrain_access("?access-2".parse().unwrap(),
                                 |a| a.shape.ndim() == 1 && a.item_shape.ndim() == 1))
}

fn is_access() -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
    move |egraph, id, _| match &egraph[id].data {
        MyAnalysisData::AccessPattern(_) => true,
        _ => false,
    }
}

fn not_item_axis(axis: Var, access: Var) -> impl Fn(&mut EG, egg::Id, &egg::Subst) -> bool {
    move |egraph, _id, subst| match &egraph[subst[access]].data {
        MyAnalysisData::AccessPattern(a) => {
            MyAnalysis::get_usize(subst[axis], egraph) < a.shape.ndim()
        }
        _ => panic!(),
    }
}

pub fn systolic_array_with_blocking(rows: usize, cols: usize) -> Rewrite<Language, MyAnalysis> {
    struct ApplierImpl {
        rows: usize,
        cols: usize,
        a: Var,
        b: Var,
    }
    impl Applier<Language, MyAnalysis> for ApplierImpl {
        fn apply_one(
            &self,
            egraph: &mut EG,
            eclass: Id,
            subst: &Subst,
            _searcher_ast: Option<&PatternAst<Language>>,
            _rule_name: Symbol,
        ) -> Vec<Id> {
            let (a, b) = match (&egraph[subst[self.a]].data, &egraph[subst[self.b]].data) {
                (MyAnalysisData::AccessPattern(a), MyAnalysisData::AccessPattern(b)) => (a, b),
                _ => panic!(),
            };
            assert_eq!(a.item_shape.ndim(), 1);
            assert_eq!(b.shape.ndim(), 1);
            assert_eq!(b.item_shape.ndim(), 1);

            let pattern: Pattern<Language> = format!(
                "(systolic-array-with-blocking {} {}
                          ?access-1
                          (access (access-transpose ?access-2 (list 1 0)) 0)
                         )",
                self.rows, self.cols
            )
            .parse()
            .unwrap();

            pattern.apply_one(egraph, eclass, subst, _searcher_ast, _rule_name)
        }
    }

    rewrite!(format!("systolic-array-with-blocking-{}-{}", rows, cols);
             "(compute dot-product
               (access-cartesian-product
                ?access-1
                ?access-2
               )
              )
             " =>
             { ApplierImpl{rows, cols, a: "?access-1".parse().unwrap(), b: "?access-2".parse().unwrap(),}}
             // Remember: the accesses look like [M] [N] and [O] [N] for an
             // MxN X NxO multiplication.
             if constrain_access("?access-1".parse().unwrap(),
                                 move |a| a.shape.ndim() <= 1 && a.item_shape.ndim() == 1
                                          && a.item_shape.slice()[0] % rows == 0)
             if constrain_access("?access-2".parse().unwrap(),
                                 move |a| a.shape.ndim() == 1 && a.item_shape.ndim() == 1
                                          && a.shape.slice()[0] % cols == 0))
}
