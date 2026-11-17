use egg::{EGraph, Id, RecExpr, Runner};
use glenside::language::from_relay::conv2d;
use glenside::language::interpreter::{interpret_from_str, Value};
use glenside::language::{rewrites, Language, MyAnalysis, MyAnalysisData};
use ndarray::{ArrayD, Dimension, IxDyn};
use std::collections::HashMap;
use std::time::Duration;

const PROGRAM: &str = r#"
(access-transpose
  (compute dot-product
    (access-cartesian-product
      (access (access-tensor weights) 1)
      (access
        (access-squeeze
          (access-windows
            (access
              (access-pad
                (access-pad (access-tensor data) zero-padding 2 0 0)
                zero-padding 3 0 0)
              1)
            (shape 1 2 2)
            (shape 1 1 1))
          1)
        3)))
  (list 1 0 2 3))
"#;

fn main() {
    let shapes = HashMap::from([
        ("data".to_owned(), vec![1, 1, 3, 3]),
        ("weights".to_owned(), vec![1, 1, 2, 2]),
    ]);
    let expression: RecExpr<Language> = PROGRAM.parse().unwrap();
    let mut constructed: RecExpr<Language> = "(access-tensor data)".parse().unwrap();
    let data = Id::from(constructed.as_ref().len() - 1);
    let name = constructed.add(Language::Symbol("weights".to_owned()));
    let weights = constructed.add(Language::AccessTensor(name));
    conv2d(&mut constructed, data, &[1, 1, 3, 3], weights, &[1, 1, 2, 2],
        &[1, 1], &[0, 0, 0, 0], &[1, 1], 1, "NCHW", "OIHW", "", false);
    assert_eq!(constructed.to_string(), expression.to_string());
    let mut graph = EGraph::new(MyAnalysis {
        name_to_shape: shapes,
        name_to_dtype: HashMap::new(),
    });
    let root = graph.add_expr(&expression);
    graph.rebuild();
    match &graph[root].data {
        MyAnalysisData::AccessPattern(access) => {
            assert_eq!(access.shape.slice(), &[1, 1, 2, 2]);
            assert!(access.item_shape.slice().is_empty());
        }
        _ => panic!("expected an access pattern"),
    }
    let tensors = HashMap::from([
        ("data", ArrayD::from_shape_vec(IxDyn(&[1, 1, 3, 3]),
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap()),
        ("weights", ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]),
            vec![1.0_f64, 0.0, 0.0, -1.0]).unwrap()),
    ]);
    match interpret_from_str(PROGRAM, &tensors) {
        Value::Access(access) => {
            let values: Vec<f64> = access.tensor.iter().copied().collect();
            assert_eq!(access.tensor.shape(), &[1, 1, 2, 2]);
            assert_eq!(access.access_axis, 4);
            assert_eq!(values, vec![-4.0; 4]);
            println!("original shape={:?}, access_axis={}, values={:?}",
                access.tensor.shape(), access.access_axis, values);
        }
        _ => panic!("expected an interpreted access"),
    }
    let initial_nodes = graph.total_size();
    let rules = vec![
        rewrites::flatten_unflatten_any_access(),
        rewrites::bubble_reshape_through_cartesian_product(),
        rewrites::bubble_reshape_through_compute_dot_product(),
        rewrites::bubble_access_concatenate_through_compute_dot_product_not_item_axis(),
        rewrites::systolic_array(),
    ];
    let runner = Runner::default()
        .with_egraph(graph)
        .with_iter_limit(30)
        .with_node_limit(100000)
        .with_time_limit(Duration::from_secs(10))
        .run(&rules);
    println!("local Runner: iterations={}, nodes_before={}, nodes_after={}",
        runner.iterations.len(), initial_nodes, runner.egraph.total_size());
}
