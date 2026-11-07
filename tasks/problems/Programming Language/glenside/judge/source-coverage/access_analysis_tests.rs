#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn access_windows() {
        // TODO(@gussmith23) Could probably clean this up with a for loop
        // Would make it easier to add more tests.

        let program = "
         (access-windows (access (access-tensor t-3-32-32) 0) (slice-shape (shape-of t-8-3-3-3) 1) (shape 1 1 1))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 30, 30]));
                assert_eq!(a.item_shape, IxDyn(&[3, 3, 3]));
            }
            _ => panic!(),
        }

        let program = "
         (access-windows (access (access-tensor t-3-32-32) 0) (slice-shape (shape-of t-8-3-3-3) 1) (shape 1 2 1))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 15, 30]));
                assert_eq!(a.item_shape, IxDyn(&[3, 3, 3]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn shape_of() {
        let program = "
         (shape-of t-3-32-32)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        assert_eq!(
            MyAnalysis::get_shape_of_value(id, &egraph),
            &IxDyn(&[3, 32, 32])
        );
    }

    #[test]
    fn access() {
        let program = "
         (access (access-tensor t-3-32-32) 0)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[3, 32, 32]));
            }
            _ => panic!(),
        }

        let program = "
         (access (access-tensor t-3-32-32) 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[32]));
            }
            _ => panic!(),
        }

        let program = "
         (access (access-tensor t-3-32-32) 3)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn reaccess() {
        let program = "
         (access (access (access-tensor t-3-32-32) 3) 0)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[3, 32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic]
    fn access_invalid() {
        let program = "
         (access (access-tensor t-3-32-32) 4)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
    }

    #[test]
    fn shape_insert_axis_0() {
        let program = "
         (shape-insert-axis (shape 1 2 3) 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        assert_eq!(
            MyAnalysis::get_shape_of_value(id, &egraph),
            &IxDyn(&[1, 2, 1, 3])
        );
    }

    #[test]
    fn shape_insert_axis_1() {
        let program = "
         (shape-insert-axis (shape 1 2 3) 3)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        assert_eq!(
            MyAnalysis::get_shape_of_value(id, &egraph),
            &IxDyn(&[1, 2, 3, 1])
        );
    }

    #[test]
    fn shape_remove_axis_0() {
        let program = "
         (shape-remove-axis (shape 1 2 3) 0)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        assert_eq!(MyAnalysis::get_shape_of_value(id, &egraph), &IxDyn(&[2, 3]));
    }

    #[test]
    fn shape_remove_axis_1() {
        let program = "
         (shape-remove-axis (shape 1 2 3) 1)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        assert_eq!(MyAnalysis::get_shape_of_value(id, &egraph), &IxDyn(&[1, 3]));
    }

    #[test]
    fn shape_remove_axis_2() {
        let program = "
         (shape-remove-axis (shape 1 2 3) 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        assert_eq!(MyAnalysis::get_shape_of_value(id, &egraph), &IxDyn(&[1, 2]));
    }

    #[test]
    #[should_panic]
    fn shape_remove_axis_panic() {
        let program = "
         (shape-remove-axis (shape 1 2 3) 3)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
    }

    #[test]
    #[should_panic]
    fn shape_insert_axis_panic() {
        let program = "
         (shape-insert-axis (shape 1 2 3) 4)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
    }

    #[test]
    fn slice_shape() {
        let program = "
         (slice-shape (shape-of t-3-32-32) 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        assert_eq!(MyAnalysis::get_shape_of_value(id, &egraph), &IxDyn(&[32]));

        let program = "
         (slice-shape (shape-of t-3-32-32) 0)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        assert_eq!(
            MyAnalysis::get_shape_of_value(id, &egraph),
            &IxDyn(&[3, 32, 32])
        );
    }

    #[test]
    #[should_panic]
    fn slice_shape_invalid_slice() {
        let program = "
         (slice-shape (shape-of t-3-32-32) 10)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        assert_eq!(MyAnalysis::get_shape_of_value(id, &egraph), &IxDyn(&[]));
    }

    #[test]
    fn access_cartesian_product() {
        let program = "
         (access-cartesian-product
          (access (access-tensor v-32) 0)
          (access (access-tensor t-32-32) 1)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[2, 32]));
            }
            _ => panic!(),
        }

        let program = "
         (access-cartesian-product
          (access (access-tensor t-32-32) 1)
          (access (access-tensor v-32) 0)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[2, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    // TODO(@gussmith23) More tests of cart prod w/ padding
    fn access_cartesian_product_zero_padding() {
        let program = "
         (access-cartesian-product
          (access-pad
           (access (access-tensor v-32) 0)
           zero-padding 0 2 3
          )
          (access-pad
           (access (access-tensor t-32-32) 1)
           zero-padding 1 2 3
          )
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[2, 37]));
                assert_eq!(a.zero_regions.len(), 1);
                assert_eq!(a.zero_regions[&2].len(), 37);
                assert!(a.zero_regions[&2].covered((0, 2)));
                assert!(!a.zero_regions[&2].covered((2, 34)));
                assert!(a.zero_regions[&2].covered((34, 37)));
                assert_eq!(
                    a.zero_regions[&2],
                    std::iter::repeat(true)
                        .take(2)
                        .chain(std::iter::repeat(false).take(32))
                        .chain(std::iter::repeat(true).take(3))
                        .collect::<Vec<_>>()
                )
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_dot_product() {
        let program = "
         (compute dot-product (access (access-tensor t-3-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }

        let program = "
         (compute dot-product (access (access-tensor t-3-32-32) 1))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }

        let program = "
         (compute dot-product (access (access-tensor t-3-32-32) 2))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    // This may not panic in the future, if we allow dot products over empty
    // tuples.
    #[should_panic]
    #[test]
    fn compute_dot_product_panic() {
        let program = "
         (compute dot-product (access (access-tensor t-3-32-32) 3))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn conv2d() {
        // The following TVM/Python code will compute the correct sizes of
        // cov2ds.
        //
        // import tvm
        // from tvm import relay
        //
        // mod = relay.module.Module.from_expr(
        //     relay.nn.conv2d(relay.var('x', shape=[1, 3, 32, 32]),
        //                     relay.var('weight', shape=[8, 3, 3, 3])))
        //
        // print(mod)

        let program = "
         (compute dot-product
          (access-cartesian-product
           (access (access-tensor t-8-3-3-3) 1)
           (access-squeeze
            (access-windows
             (access (access-tensor t-3-32-32) 0)
             (shape 3 3 3)
             (shape 1 1 1)
            )
            0
           )
          )
         )
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);

        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[8, 30, 30]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }

        let program = "
         (compute dot-product
          (access-cartesian-product
           (access (access-tensor t-8-3-3-3) 1)
           (access-squeeze
            (access-windows
             (access (access-tensor t-3-32-32) 0)
             (shape 3 3 3)
             (shape 1 1 2)
            )
            0
           )
          )
         )
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);

        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[8, 30, 15]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn flatten_reshape() {
        let program = "
         (access-reshape
          (access-flatten (access (access-tensor t-3-32-32) 2))
          (access-shape (shape 32 3) (shape 16 2))
         )
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);

        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 3]));
                assert_eq!(a.item_shape, IxDyn(&[16, 2]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn flatten_reshape_panic() {
        let program = "
         (access-reshape
          (access-flatten (access (access-tensor t-3-32-32) 2))
          (access-shape (shape 1) (shape 16 2))
         )
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);

        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 3]));
                assert_eq!(a.item_shape, IxDyn(&[16, 2]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_slice_0() {
        let program = "(access-slice (access (access-tensor t-3-32-32) 1) 0 0 1)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_slice_1() {
        let program = "(access-slice (access (access-tensor t-3-32-32) 1) 1 16 32)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[16, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_slice_2() {
        let program = "(access-slice (access (access-tensor t-3-32-32) 2) 2 16 32)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[16]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_slice_zero_pad_0() {
        test_logger::ensure_env_logger_initialized();

        let program = "(access-slice (access-pad (access (access-tensor t-3-32-32) 1) zero-padding 0 2 3) 0 0 3)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
                assert_eq!(a.zero_regions.len(), 1);
                assert_eq!(a.zero_regions[&0].len(), 3);
                assert_eq!(a.zero_regions[&0], vec![true, true, false]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_slice_zero_pad_1() {
        test_logger::ensure_env_logger_initialized();

        let program = "
(access-slice
 (access-pad
  (access (access-tensor t-3-32-32) 1)
  zero-padding 0 2 3
 )
 0 1 7
)"
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[6]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
                assert_eq!(a.zero_regions.len(), 1);
                assert_eq!(a.zero_regions[&0].len(), 6);
                assert_eq!(
                    a.zero_regions[&0],
                    vec![true, false, false, false, true, true]
                );
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic]
    fn access_slice_panic() {
        let program = "(access-slice (access (access-tensor t-3-32-32) 1) 3 16 32)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[16, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_concatenate_0() {
        let program = "(access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-3-32-32) 1) 0)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[6]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_concatenate_1() {
        let program = "(access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-3-32-32) 1) 2)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32, 64]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn access_concatenate_panic_0() {
        let program = "(access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-3-32-32) 1) 3)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32, 64]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn access_concatenate_panic_1() {
        let program = "(access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-8-3-3-3) 1) 2)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32, 64]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_transpose_0() {
        let program = "(access-transpose (access (access-tensor t-3-32-32) 1) (list 1 2 0))"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[32, 3]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_transpose_4() {
        let program = "(access-transpose (access (access-tensor t-3-32-32) 1) (list 1 0 2))"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[3, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_transpose_5() {
        let program = "(access-transpose (access (access-tensor t-3-32-32) 1) (list 0 1 2))"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn access_transpose_panic_2() {
        let program = "(access-transpose (access (access-tensor t-3-32-32) 1) (list 0 1 3))"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn access_move_axis_panic_1() {
        let program = "(access-move-axis (access (access-tensor t-3-32-32) 1) 1 3)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_reduce_sum_0() {
        let program = "
         (compute reduce-sum (access (access-tensor t-3-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_reduce_sum_1() {
        let program = "
         (compute reduce-sum (access (access-tensor t-3-32-32) 2))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_pair() {
        let program = "
         (access-pair (access (access-tensor t-32-32) 1) (access (access-tensor t-32-32) 1))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[2, 32]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn access_pair_panic() {
        let program = "
         (access-pair (access (access-tensor t-32-32) 0) (access (access-tensor t-32-32) 1))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[2, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_shift_right_0() {
        let program = "
         (access-shift-right (access (access-tensor t-32-32) 2))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_shift_right_1() {
        let program = "
         (access-shift-right (access (access-tensor t-32-32) 1))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_shift_right_2() {
        let program = "
         (access-shift-right (access (access-tensor t-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_relu() {
        let program = "
         (compute relu (access (access-tensor t-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_elementwise_add_0() {
        let program = "
         (compute elementwise-add (access (access-tensor t-3-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_elementwise_add_1() {
        let program = "
         (compute elementwise-add (access (access-tensor t-3-32-32) 1))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_elementwise_add_2() {
        let program = "
         (compute elementwise-add (access (access-tensor t-3-32-32) 2))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn compute_elementwise_add_panic() {
        let program = "
         (compute elementwise-add (access (access-tensor t-3-32-32) 3))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
    }

    #[test]
    fn compute_elementwise_mul_0() {
        let program = "
         (compute elementwise-mul (access (access-tensor t-3-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_elementwise_mul_1() {
        let program = "
         (compute elementwise-mul (access (access-tensor t-3-32-32) 1))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_elementwise_mul_2() {
        let program = "
         (compute elementwise-mul (access (access-tensor t-3-32-32) 2))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn compute_elementwise_mul_panic() {
        let program = "
         (compute elementwise-mul (access (access-tensor t-3-32-32) 3))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
    }

    #[test]
    fn zero_padding() {
        let program = "zero-padding".parse().unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::PadType(PadType::ZeroPadding) => (),
            _ => panic!(),
        };
    }

    #[test]
    fn access_pad_zero_padding_0() {
        let program = "
         (access-pad (access (access-tensor t-32-32) 1) zero-padding 0 1 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[35]));
                assert_eq!(a.item_shape, IxDyn(&[32]));
                assert_eq!(a.zero_regions.len(), 1);
                assert_eq!(a.zero_regions[&0].len(), 35);
                assert!(a.zero_regions[&0].covered((0, 1)));
                assert!(!a.zero_regions[&0].covered((1, 33)));
                assert!(a.zero_regions[&0].covered((33, 35)));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_pad_zero_padding_1() {
        let program = "
         (access-pad (access (access-tensor t-32-32) 1) zero-padding 1 0 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[34]));
                assert_eq!(a.zero_regions.len(), 1);
                assert_eq!(a.zero_regions[&1].len(), 34);
                assert!(a.zero_regions[&1].covered((32, 34)));
                assert!(!a.zero_regions[&1].covered((0, 32)));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_pad_zero_padding_2() {
        let program = "
(access-pad
 (access-pad (access (access-tensor t-32-32) 1) zero-padding 1 0 2)
 zero-padding 0 1 3
)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[36]));
                assert_eq!(a.item_shape, IxDyn(&[34]));
                assert_eq!(a.zero_regions.len(), 2);
                assert_eq!(a.zero_regions[&1].len(), 34);
                assert_eq!(a.zero_regions[&0].len(), 36);
                assert!(a.zero_regions[&1].covered((32, 34)));
                assert!(!a.zero_regions[&1].covered((0, 32)));
                assert!(a.zero_regions[&0].covered((0, 1)));
                assert!(a.zero_regions[&0].covered((33, 36)));
                assert!(!a.zero_regions[&0].covered((1, 33)));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_pad_zero_padding_3() {
        let program = "
(access-pad
 (access-pad (access (access-tensor t-32-32) 1) zero-padding 0 0 2)
 zero-padding 0 1 3
)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[38]));
                assert_eq!(a.item_shape, IxDyn(&[32]));
                assert_eq!(a.zero_regions.len(), 1);
                assert_eq!(a.zero_regions[&0].len(), 38);
                assert!(a.zero_regions[&0].covered((0, 1)));
                assert!(a.zero_regions[&0].covered((35, 38)));
                assert!(!a.zero_regions[&0].covered((1, 35)));
                // This one is key: this makes sure that the first pad's zero
                // region was shifted appropriately by the second pad (was (32,
                // 34), but should get shifted by 1)
                assert!(a.zero_regions[&0].covered((33, 35)));
                assert!(!a.zero_regions[&0].covered((0, 33)));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic]
    fn access_pad_zero_padding_panic() {
        let program = "
         (access-pad (access (access-tensor t-32-32) 1) zero-padding 2 3 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
    }

    #[test]
    fn compute_reduce_max_0() {
        let program = "
         (compute reduce-max (access (access-tensor t-3-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_reduce_max_1() {
        let program = "
         (compute reduce-max (access (access-tensor t-3-32-32) 2))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_squeeze() {
        let program = "
         (access-squeeze (access (access-tensor t-1-2-3-4) 1) 0)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[2, 3, 4]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic]
    fn access_squeeze_panic() {
        let program = "
         (access-squeeze (access (access-tensor t-1-2-3-4) 1) 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
    }

    #[test]
    fn access_insert_axis() {
        let program = "
         (access-insert-axis (access (access-tensor t-1-2-3-4) 1) 0)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 1]));
                assert_eq!(a.item_shape, IxDyn(&[2, 3, 4]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic]
    // TODO(@gussmith) More access-insert-axis tests
    fn access_insert_axis_panic() {
        let program = "
         (access-squeeze (access (access-tensor t-1-2-3-4) 1) 5)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
    }

    #[test]
    // TODO(@gussmith) More access-broadcast tests
    fn access_broadcast() {
        let program = "
         (access-broadcast (access (access-tensor t-1-2-3-4) 1) (access-shape (shape 2 2 3 4) (shape)))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: HashMap::default(),
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[2]));
                assert_eq!(a.item_shape, IxDyn(&[2, 3, 4]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn systolic_array() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        let program = "
         (systolic-array 64 32
          (access (access-tensor a) 1)
          (access (access-transpose (access-tensor a) (list 1 0)) 0)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic]
    fn systolic_array_panic_0() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        // Because the second argument is not the right shape.
        let program = "
         (systolic-array 64 32
          (access (access-tensor a) 1)
          (access (access-tensor a) 0)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        egraph.add_expr(&program);
    }

    #[test]
    fn list() {
        let program = "
         (list 1 2 3 4)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::List(l) => assert_eq!(l, &vec![1, 2, 3, 4]),
            _ => panic!(),
        }
    }

    #[test]
    fn access_transpose() {
        let program = "
         (access-transpose (access (access-tensor a) 1) (list 2 0 1))
         "
        .parse()
        .unwrap();
        let mut map = HashMap::default();
        let name_to_dtype = [("a".into(), DataType::Float(32))]
            .iter()
            .cloned()
            .collect();
        map.insert("a".to_string(), vec![4, 5, 6]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype,
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[6]));
                assert_eq!(a.item_shape, IxDyn(&[4, 5]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_transpose_1() {
        let program = "
             (access-transpose
              (access-transpose
               (access (access-tensor t) 1)
               (list 1 3 2 0)
              )
              (list 3 2 1 0)
             )"
        .parse()
        .unwrap();
        let mut map = HashMap::new();
        let name_to_dtype = [("t".into(), DataType::Float(32))]
            .iter()
            .cloned()
            .collect();
        map.insert("t".to_string(), vec![1, 2, 3, 4]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype,
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1]));
                assert_eq!(a.item_shape, IxDyn(&[3, 4, 2]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_transpose_2() {
        let program = "
              (access-transpose
               (access (access-tensor t) 1)
               (list 1 3 2 0)
              )
             "
        .parse()
        .unwrap();
        let mut map = HashMap::new();
        map.insert("t".to_string(), vec![1, 2, 3, 4]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[2]));
                assert_eq!(a.item_shape, IxDyn(&[4, 3, 1]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_transpose_3() {
        let program = "
              (access-transpose
               (access-pad (access (access-tensor t) 1) zero-padding 1 5 0)
               (list 1 3 2 0)
              )
             "
        .parse()
        .unwrap();
        let mut map = HashMap::new();
        map.insert("t".to_string(), vec![1, 2, 3, 4]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[7]));
                assert_eq!(a.item_shape, IxDyn(&[4, 3, 1]));
                assert_eq!(a.zero_regions.len(), 1);
                assert_eq!(a.zero_regions[&0].len(), 7);
                assert!(a.zero_regions[&0].covered((0, 5)));
                assert!(!a.zero_regions[&0].covered((5, 7)));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic]
    fn access_transpose_panic_0() {
        let program = "
         (access-transpose (access (access-tensor a) 1) (list 0 1))
         "
        .parse()
        .unwrap();
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![4, 5, 6]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        egraph.add_expr(&program);
    }

    #[test]
    #[should_panic]
    fn access_transpose_panic_1() {
        let program = "
         (access-transpose (access (access-tensor a) 1) (list 2 1 1))
         "
        .parse()
        .unwrap();
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![4, 6]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        egraph.add_expr(&program);
    }

    #[test]
    #[should_panic]
    fn compute_softmax_0() {
        let program = "
         (compute softmax (access (access-tensor t-3-32-32) 3))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
    }

    #[test]
    #[should_panic]
    fn compute_softmax_1() {
        let program = "
         (compute softmax (access (access-tensor t-3-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
    }

    #[test]
    fn compute_softmax_2() {
        let program = "
         (compute softmax (access (access-tensor t-3-32-32) 2))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_reduce_mean_0() {
        let program = "
         (compute reduce-mean (access (access-tensor t-3-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_reduce_mean_1() {
        let program = "
         (compute reduce-mean (access (access-tensor t-3-32-32) 2))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_pad_min_padding() {
        let program = "
         (access-pad (access (access-tensor t-32-32) 1) min-padding 0 1 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[35]));
                assert_eq!(a.item_shape, IxDyn(&[32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_elementwise_div() {
        let program = "
         (compute elementwise-div (access (access-tensor t-3-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn literal_0() {
        let program = "
         (literal 0.1234)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::Literal(t) => {
                assert_eq!(*t, ndarray::arr0(0.1234).into_dyn());
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_literal() {
        let program = "
         (access-literal (literal 0.1234))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_sqrt() {
        let program = "
         (compute sqrt (access (access-tensor t-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_negative() {
        let program = "
         (compute negative (access (access-tensor t-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn systolic_array_with_blocking_0() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        map.insert("b".to_string(), vec![64, 32]);
        let program = "
         (systolic-array-with-blocking 64 32
          (access (access-tensor a) 1)
          (access (access-tensor b) 0)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn systolic_array_with_blocking_2() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        map.insert("b".to_string(), vec![64, 32]);
        let program = "
         (systolic-array-with-blocking 32 32
          (access (access-tensor a) 1)
          (access (access-tensor b) 0)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn systolic_array_with_blocking_3() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        map.insert("b".to_string(), vec![64, 32]);
        let program = "
         (systolic-array-with-blocking 32 2
          (access (access-tensor a) 1)
          (access (access-tensor b) 0)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic(expected = "assertion failed: a1.item_shape.slice()[1] % cols == 0")]
    fn systolic_array_with_blocking_panic() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        map.insert("b".to_string(), vec![64, 32]);
        let program = "
         (systolic-array-with-blocking 32 3
          (access (access-tensor a) 1)
          (access (access-tensor b) 0)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let _id = egraph.add_expr(&program);
    }

    #[test]
    fn batch_norm_inference() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        // Just showing that, right now, these shapes don't matter at all.
        map.insert("b".to_string(), vec![32]);
        map.insert("c".to_string(), vec![10, 20]);
        map.insert("d".to_string(), vec![3, 3, 3, 3, 3, 3]);
        map.insert("e".to_string(), vec![]);
        let program = "
         (relay-operator-call relay-batch-norm-inference
          (access-tensor a)
          (access-tensor b)
          (access-tensor c)
          (access-tensor d)
          (access-tensor e)
          1 1e-5)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic(expected = "Parameters do not type check")]
    fn batch_norm_inference_panic() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        // Just showing that, right now, these shapes don't matter at all.
        map.insert("b".to_string(), vec![32]);
        map.insert("c".to_string(), vec![10, 20]);
        map.insert("d".to_string(), vec![3, 3, 3, 3, 3, 3]);
        map.insert("e".to_string(), vec![]);
        let program = "
         (relay-operator-call relay-batch-norm-inference
          (access-tensor a)
          (access-tensor b)
          (access-tensor c)
          (access-tensor d)
          (access-tensor e)
          1e-5 1)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_softmax() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        let program = "
         (relay-operator-call relay-softmax
          (access-tensor a)
          1)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic(expected = "Parameters do not type check")]
    fn relay_operator_call_softmax_panic() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        map.insert("b".to_string(), vec![32]);
        let program = "
         (relay-operator-call relay-softmax
          (access-tensor a)
          (access-tensor b)
          )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_relu() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        let program = "
         (relay-operator-call relay-relu
          (access-tensor a))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic(expected = "Parameters do not type check")]
    fn relay_operator_call_relu_panic() {
        let program = "
         (relay-operator-call relay-relu 1)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: HashMap::default(),
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_max_pool2d_nchw() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 3, 32, 64]);
        let program = "
         (relay-operator-call relay-max-pool2d
          (access-tensor a)
          (shape 1 2)
          (shape 3 4)
          (shape 5 6 7 8)
          relay-activation-layout-nchw
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 3, 15, 20]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_max_pool2d_nhwc() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 3, 32, 64]);
        let program = "
         (relay-operator-call relay-max-pool2d
          (access-tensor a)
          (shape 1 2)
          (shape 3 4)
          (shape 5 6 7 8)
          relay-activation-layout-nhwc
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 5, 12, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic(expected = "Parameters do not type check")]
    fn relay_operator_call_max_pool2d_panic() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 3, 32, 64]);
        let program = "
         (relay-operator-call relay-max-pool2d
          (access-tensor a)
          (access-tensor a)
          (shape 3 4)
          (shape 5 6 7 8)
          relay-activation-layout-nhwc
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_global_avg_pool2d_nchw() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 3, 32, 64]);
        let program = "
         (relay-operator-call relay-global-avg-pool2d
          (access-tensor a)
          relay-activation-layout-nchw
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 3, 1, 1]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_global_avg_pool2d_nhwc() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 3, 32, 64]);
        let program = "
         (relay-operator-call relay-global-avg-pool2d
          (access-tensor a)
          relay-activation-layout-nhwc
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 1, 1, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic(expected = "Parameters do not type check")]
    fn relay_operator_call_global_avg_pool2d_panic() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 3, 32, 64]);
        let program = "
         (relay-operator-call relay-max-pool2d
          (access-tensor a)
          (access-tensor a)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_batch_flatten() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 3, 32, 64]);
        let program = "
         (relay-operator-call relay-batch-flatten
          (access-tensor a)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 3 * 32 * 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic(expected = "Parameters do not type check")]
    fn relay_operator_call_batch_flatten_panic() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 3, 32, 64]);
        let program = "
         (relay-operator-call relay-batch-flatten
          (access-tensor a)
          (access-tensor a)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_bias_add() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 32]);
        map.insert("b".to_string(), vec![32]);
        let program = "
         (relay-operator-call relay-bias-add
          (access-tensor a)
          (access-tensor b)
          1
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic(expected = "Parameters do not type check")]
    fn relay_operator_call_bias_add_panic() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 32]);
        let program = "
         (relay-operator-call relay-bias-add
          (access-tensor a)
          2 3
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_add() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![3, 1, 32, 32]);
        map.insert("b".to_string(), vec![64, 1, 32]);
        let program = "
         (relay-operator-call relay-add
          (access-tensor a)
          (access-tensor b)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 64, 32, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic(expected = "Shapes can't be broadcast")]
    fn relay_operator_call_add_panic_broadcast() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![3, 2, 32, 32]);
        map.insert("b".to_string(), vec![64, 1, 32]);
        let program = "
         (relay-operator-call relay-add
          (access-tensor a)
          (access-tensor b)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 64, 32, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic(expected = "Parameters do not type check")]
    fn relay_operator_call_add_panic_types() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![3, 2, 32, 32]);
        let program = "
         (relay-operator-call relay-add
          (access-tensor a)
          1
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 64, 32, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_leaky_relu() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        let program = "
         (relay-operator-call relay-leaky-relu
          (access-tensor a) 0.1)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_avgpool2d() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 1280, 7, 7]);
        let program = "
         (relay-operator-call relay-avg-pool2d
          (access-tensor a)
          (shape 7 7)
          (shape 1 1)
          (shape 0 0 0 0)
          relay-activation-layout-nchw)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 1280, 1, 1]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_upsampling() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 1280, 7, 7]);
        let program = "
         (relay-operator-call relay-upsampling
          (access-tensor a)
          2.0 2.0
          (relay-activation-layout-nchw))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 1280, 14, 14]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_sigmoid() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 1280, 7, 7]);
        let program = "
         (relay-operator-call relay-sigmoid
          (access-tensor a))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 1280, 7, 7]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_split() {
        let names_to_shapes = [("data".into(), vec![1, 5, 4])]
            .iter()
            .cloned()
            .collect::<HashMap<_, _>>();
        let mut program = egg::RecExpr::default();
        let operator_id = program.add(Language::RelayOperator(RelayOperator::RelaySplit));
        let tensor_id = program.add(Language::Symbol("data".into()));
        let access_data = program.add(Language::AccessTensor(tensor_id));
        let sections = program.add(Language::Num(5));
        let axis = program.add(Language::Num(1));
        let _relay_operator_call = program.add(Language::RelayOperatorCall(
            vec![operator_id, access_data, sections, axis].into_boxed_slice(),
        ));
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: names_to_shapes,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        let section_data = MyAnalysisData::AccessPattern(AccessPatternData {
            shape: IxDyn(&[]),
            item_shape: IxDyn(&[1, 1, 4]),
            access_pattern_shape_settled: false,
            zero_regions: HashMap::default(),
            contains_accelerator_calls: false,
        });
        match &egraph[id].data {
            MyAnalysisData::Tuple(tup) => tup
                .iter()
                .zip(vec![section_data.clone(), section_data.clone(), section_data].iter())
                .for_each(|t| assert_eq!(t.0, t.1)),
            _ => panic!("Split should outputs a tuple"),
        }
    }

    #[test]
    fn relay_operator_call_maximum() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 1280, 7, 7]);
        map.insert("b".to_string(), vec![1, 1280, 1, 1]);
        let program = "
         (relay-operator-call relay-maximum
          (access-tensor a) (access-tensor b))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 1280, 7, 7]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_minimum() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 1280, 7, 7]);
        map.insert("b".to_string(), vec![1, 1280, 1, 1]);
        let program = "
         (relay-operator-call relay-minimum
          (access-tensor a) (access-tensor b))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 1280, 7, 7]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn systolic_array_conv2d_nchw_oihw_with_blocking() {
        let mut map = HashMap::default();
        map.insert("data".to_string(), vec![1, 32, 44, 78]);
        map.insert("weights".to_string(), vec![64, 32, 1, 2]);
        let program = "
         (systolic-array-conv2d-nchw-oihw-with-blocking
          32 32
          (access-tensor weights)
          (access-tensor data)
          1 2
          3 4
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 64, 15, 20]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn systolic_array_conv2d_nhwc_hwio_with_blocking() {
        let mut map = HashMap::default();
        map.insert("data".to_string(), vec![1, 44, 78, 32]);
        map.insert("weights".to_string(), vec![1, 2, 32, 64]);
        let program = "
         (systolic-array-conv2d-nhwc-hwio-with-blocking
          32 32
          (access-tensor weights)
          (access-tensor data)
          1 2
          3 4
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 15, 20, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn systolic_array_conv2d_im2col_nhwc_hwio_with_blocking() {
        let mut map = HashMap::default();
        // Note that we don't need any multiples of rows/cols here.
        // Systolic array should handle tail padding.
        map.insert("data".to_string(), vec![1, 44, 78, 33]);
        map.insert("weights".to_string(), vec![1, 2, 33, 65]);
        let program = "
         (systolic-array-conv2d-im2col-nhwc-hwio-with-blocking
          32 32
          (access-tensor weights)
          (access-tensor data)
          1 2
          3 4
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 15, 20, 65]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn systolic_array_conv2d_im2col_nchw_oihw_with_blocking() {
        let mut map = HashMap::default();
        // Note that we don't need any multiples of rows/cols here.
        // Systolic array should handle tail padding.
        map.insert("data".to_string(), vec![1, 33, 44, 78]);
        map.insert("weights".to_string(), vec![65, 33, 1, 2]);
        let program = "
         (systolic-array-conv2d-im2col-nchw-oihw-with-blocking
          32 32
          (access-tensor weights)
          (access-tensor data)
          1 2
          3 4
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 65, 15, 20]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn construct_tuple() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        map.insert("b".to_string(), vec![16, 32]);
        let program = "
         (construct-tuple (access-tensor a) (access-tensor b))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::Tuple(a) => {
                assert_eq!(a.len(), 2);
                match (a[0].clone(), a[1].clone()) {
                    (MyAnalysisData::AccessPattern(t1), MyAnalysisData::AccessPattern(t2)) => {
                        assert_eq!(t1.shape, IxDyn(&[32, 64]));
                        assert_eq!(t2.shape, IxDyn(&[16, 32]));
                    }
                    _ => panic!(),
                }
            }
            _ => panic!(),
        }
    }

    #[test]
    fn tuple_get_item() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        map.insert("b".to_string(), vec![16, 32]);
        let program = "
         (tuple-get-item (construct-tuple (access-tensor a) (access-tensor b)) 1)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(b) => {
                assert_eq!(b.shape, IxDyn(&[16, 32]));
            }
            _ => panic!(),
        }
    }

    // >>> data = relay.var('data', shape=(2, 3, 32, 32))
    // >>> weights = relay.var('weights', shape=(3, 1, 5, 5))
    // >>> program = relay.nn.conv2d(data, weights, strides=(2, 3), padding=(1, 2, 3, 4), groups=3, channels=3)
    // >>> mod = tvm.IRModule.from_expr(program)
    // >>> program = relay.nn.conv2d(data, weights, strides=(2, 3), padding=(1, 2, 3, 4), groups=3, channels=3)
    // >>> mod = relay.transform.InferType()(mod)
    // >>> mod
    // #[version = "0.0.5"]
    // def @main(%data: Tensor[(2, 3, 32, 32), float32], %weights: Tensor[(3, 1, 5, 5), float32]) -> Tensor[(2, 3, 16, 12), float32] {
    //   nn.conv2d(%data, %weights, strides=[2, 3], padding=[1, 2, 3, 4], groups=3, channels=3) /* ty=Tensor[(2, 3, 16, 12), float32] */
    // }
    #[test]
    fn conv2d_depthwise_0() {
        let mut map = HashMap::default();
        map.insert("data".to_string(), vec![2, 3, 32, 32]);
        map.insert("weights".to_string(), vec![3, 1, 5, 5]);

        let program = "
         (relay-operator-call relay-conv2d
            (access-tensor data)
            (access-tensor weights)
            (shape 2 3)
            (shape 1 2 3 4)
            3
            3
            (shape 3 5 5)
            relay-activation-layout-nchw
            relay-kernel-layout-oihw
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(b) => {
                assert_eq!(b.shape, IxDyn(&[2, 3, 16, 12]));
            }
            _ => panic!(),
        }
    }
    // >>> import tvm
    // >>> from tvm import relay
    // >>> data = relay.var('data', shape=(2, 3, 32, 32))
    // >>> weights = relay.var('weights', shape=(3, 1, 5, 5))
    // >>> program = relay.nn.conv2d(data, weights, strides=(4, 1), padding=(0, 2, 1, 5), groups=3, channels=3)
    // >>> mod = tvm.IRModule.from_expr(program)
    // >>> mod = relay.transform.InferType()(mod)
    // >>> mod
    // #[version = "0.0.5"]
    // def @main(%data: Tensor[(2, 3, 32, 32), float32], %weights: Tensor[(3, 1, 5, 5), float32]) -> Tensor[(2, 3, 8, 35), float32] {
    //   nn.conv2d(%data, %weights, strides=[4, 1], padding=[0, 2, 1, 5], groups=3, channels=3) /* ty=Tensor[(2, 3, 8, 35), float32] */
    // }
    #[test]
    fn conv2d_depthwise_1() {
        let mut map = HashMap::default();
        map.insert("data".to_string(), vec![2, 3, 32, 32]);
        map.insert("weights".to_string(), vec![3, 1, 5, 5]);

        let program = "
         (relay-operator-call relay-conv2d
            (access-tensor data)
            (access-tensor weights)
            (shape 4 1)
            (shape 0 2 1 5)
            3
            3
            (shape 3 5 5)
            relay-activation-layout-nchw
            relay-kernel-layout-oihw
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(b) => {
                assert_eq!(b.shape, IxDyn(&[2, 3, 8, 35]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn test_relay_cast() {
        let mut map = HashMap::default();
        map.insert("data".to_string(), vec![2, 3, 32, 32]);
        let dtypes = [("data".into(), crate::language::DataType::Int(32))]
            .iter()
            .cloned()
            .collect();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: dtypes,
        });
        let program = "
    (relay-operator-call relay-cast data float32)
    ";
        let id = egraph.add_expr(&program.parse().unwrap());
        match &egraph[id].data {
            MyAnalysisData::Shape(shape) => {
                assert_eq!(shape.dtype, crate::language::DataType::Float(32))
            }
            _ => panic!("Not a valid cast"),
        }
    }
    #[test]
    fn relay_take_0() {
        let mut map = HashMap::default();
        map.insert("data".to_string(), vec![5, 6, 7]);
        map.insert("indices".to_string(), vec![2, 3]);
        let dtypes = [
            ("data".into(), crate::language::DataType::Float(32)),
            ("indices".into(), crate::language::DataType::Int(32)),
        ]
        .iter()
        .cloned()
        .collect();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: dtypes,
        });
        let program = "
    (relay-operator-call relay-take (access-tensor data) (access-tensor indices) 0)
    ";
        let id = egraph.add_expr(&program.parse().unwrap());
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape.slice(), &[2, 3, 6, 7]);
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    /// def @main(%data: Tensor[(5, 6, 7), float32], %indices: Tensor[(2, 3), int32]) -> Tensor[(5, 6, 2, 3), float32] {
    ///   take(%data, %indices, axis=-1) /* ty=Tensor[(5, 6, 2, 3), float32] */
    /// }
    #[test]
    fn relay_take_1() {
        let mut map = HashMap::default();
        map.insert("data".to_string(), vec![5, 6, 7]);
        map.insert("indices".to_string(), vec![2, 3]);
        let dtypes = [
            ("data".into(), crate::language::DataType::Float(32)),
            ("indices".into(), crate::language::DataType::Int(32)),
        ]
        .iter()
        .cloned()
        .collect();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: dtypes,
        });
        let program = "
    (relay-operator-call relay-take (access-tensor data) (access-tensor indices) 1)
    ";
        let id = egraph.add_expr(&program.parse().unwrap());
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape.slice(), &[5, 2, 3, 7]);
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    /// def @main(%data: Tensor[(5, 6, 7), float32], %indices: Tensor[(2, 3), int32]) -> Tensor[(5, 6, 2, 3), float32] {
    ///   take(%data, %indices, axis=2) /* ty=Tensor[(5, 6, 2, 3), float32] */
    /// }
    #[test]
    fn relay_take_2() {
        let mut map = HashMap::default();
        map.insert("data".to_string(), vec![5, 6, 7]);
        map.insert("indices".to_string(), vec![2, 3]);
        let dtypes = [
            ("data".into(), crate::language::DataType::Float(32)),
            ("indices".into(), crate::language::DataType::Int(32)),
        ]
        .iter()
        .cloned()
        .collect();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: dtypes,
        });
        let program = "
    (relay-operator-call relay-take (access-tensor data) (access-tensor indices) 2)
    ";
        let id = egraph.add_expr(&program.parse().unwrap());
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape.slice(), &[5, 6, 2, 3]);
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_stack_0() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![5, 6, 7]);
        map.insert("b".to_string(), vec![5, 6, 7]);
        map.insert("c".to_string(), vec![5, 6, 7]);
        let dtypes = [
            ("a".into(), crate::language::DataType::Float(32)),
            ("b".into(), crate::language::DataType::Float(32)),
            ("c".into(), crate::language::DataType::Float(32)),
        ]
        .iter()
        .cloned()
        .collect();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: dtypes,
        });
        let program = "
    (relay-operator-call relay-stack (access-tensor a) (access-tensor b) (access-tensor c) 0)
    ";
        let id = egraph.add_expr(&program.parse().unwrap());
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape.slice(), &[3, 5, 6, 7]);
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_stack_1() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![5, 6, 7]);
        map.insert("b".to_string(), vec![5, 6, 7]);
        map.insert("c".to_string(), vec![5, 6, 7]);
        let dtypes = [
            ("a".into(), crate::language::DataType::Float(32)),
            ("b".into(), crate::language::DataType::Float(32)),
            ("c".into(), crate::language::DataType::Float(32)),
        ]
        .iter()
        .cloned()
        .collect();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: dtypes,
        });
        let program = "
    (relay-operator-call relay-stack (access-tensor a) (access-tensor b) (access-tensor c) 1)
    ";
        let id = egraph.add_expr(&program.parse().unwrap());
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape.slice(), &[5, 3, 6, 7]);
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }
    #[test]
    fn relay_stack_2() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![5, 6, 7]);
        map.insert("b".to_string(), vec![5, 6, 7]);
        map.insert("c".to_string(), vec![5, 6, 7]);
        let dtypes = [
            ("a".into(), crate::language::DataType::Float(32)),
            ("b".into(), crate::language::DataType::Float(32)),
            ("c".into(), crate::language::DataType::Float(32)),
        ]
        .iter()
        .cloned()
        .collect();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: dtypes,
        });
        let program = "
    (relay-operator-call relay-stack (access-tensor a) (access-tensor b) (access-tensor c) 2)
    ";
        let id = egraph.add_expr(&program.parse().unwrap());
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape.slice(), &[5, 6, 3, 7]);
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }
    #[test]
    fn relay_stack_3() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![5, 6, 7]);
        map.insert("b".to_string(), vec![5, 6, 7]);
        map.insert("c".to_string(), vec![5, 6, 7]);
        let dtypes = [
            ("a".into(), crate::language::DataType::Float(32)),
            ("b".into(), crate::language::DataType::Float(32)),
            ("c".into(), crate::language::DataType::Float(32)),
        ]
        .iter()
        .cloned()
        .collect();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: dtypes,
        });
        let program = "
    (relay-operator-call relay-stack (access-tensor a) (access-tensor b) (access-tensor c) 3)
    ";
        let id = egraph.add_expr(&program.parse().unwrap());
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape.slice(), &[5, 6, 7, 3]);
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }
    #[test]
    fn relay_stack_4() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![5, 6, 7]);
        map.insert("b".to_string(), vec![5, 6, 7]);
        map.insert("c".to_string(), vec![5, 6, 7]);
        let dtypes = [
            ("a".into(), crate::language::DataType::Float(32)),
            ("b".into(), crate::language::DataType::Float(32)),
            ("c".into(), crate::language::DataType::Float(32)),
        ]
        .iter()
        .cloned()
        .collect();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: dtypes,
        });
        let program = "
    (relay-operator-call relay-stack (access-tensor a) (access-tensor b) (access-tensor c) -1)
    ";
        let id = egraph.add_expr(&program.parse().unwrap());
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape.slice(), &[5, 6, 7, 3]);
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }
}
