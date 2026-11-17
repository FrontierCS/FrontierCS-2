use super::*;
use glenside::language::AccessPatternData;
use ndarray::Axis;

pub(super) fn registration_check() -> serde_json::Value {
    let workload = matrix_workload(&[2, 4], &[3, 4]);
    let expression: RecExpr<Language> = "(compute dot-product (access-cartesian-product (access (access-tensor x) 1) (access (access-tensor y) 1)))".parse().unwrap();
    let mut graph = EGraph::new(analysis(&workload));
    graph.add_expr(&expression);
    let runner = Runner::<_, _, ()>::new(MyAnalysis::default()).with_egraph(graph)
        .with_iter_limit(3).run(&trusted_mapping_rules());
    let ordinary = runner.egraph.classes().flat_map(|class| &class.nodes).any(|node| matches!(node, Language::SystolicArray(_)));
    let blocked = runner.egraph.classes().flat_map(|class| &class.nodes).any(|node| matches!(node, Language::SystolicArrayWithBlocking(_)));
    assert!(ordinary && blocked);
    let mut arbitrary: Rewrite<Language, MyAnalysis> = egg::rewrite!("arbitrary-construction";
        "(compute dot-product (access-cartesian-product ?left ?right))" =>
        "(systolic-array-with-blocking 1 1 ?left (access (access-transpose ?right (list 1 0)) 0))");
    arbitrary.applier = Arc::new(GenericApplier(arbitrary.applier.clone()));
    let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut graph = EGraph::new(analysis(&workload));
        graph.add_expr(&expression);
        Runner::<_, _, ()>::new(MyAnalysis::default()).with_egraph(graph).with_iter_limit(3).run(&[arbitrary]);
    })).is_err();
    assert!(rejected);
    serde_json::json!({"ordinary": ordinary, "blocked": blocked, "generic_rejected": rejected})
}

fn canonical_fact(data: &MyAnalysisData) -> serde_json::Value {
    match data {
        MyAnalysisData::AccessPattern(access) => serde_json::json!({
            "variant": "AccessPattern", "shape": access.shape.slice(),
            "item_shape": access.item_shape.slice(), "zero_regions": access.zero_regions,
            "settled": access.access_pattern_shape_settled,
            "accelerator": access.contains_accelerator_calls
        }),
        MyAnalysisData::Tuple(items) => serde_json::json!({"variant": "Tuple", "items": items.iter().map(canonical_fact).collect::<Vec<_>>()}),
        MyAnalysisData::Literal(value) => serde_json::json!({"variant": "Literal", "shape": value.shape(), "values": value.iter().copied().collect::<Vec<_>>()}),
        MyAnalysisData::Num(value) => serde_json::json!({"variant": "Num", "value": value}),
        MyAnalysisData::DataType(value) => serde_json::json!({"variant": "DataType", "value": value.to_string()}),
        MyAnalysisData::Shape(value) => serde_json::json!({"variant": "Shape", "shape": value.shape.slice()}),
        MyAnalysisData::List(value) => serde_json::json!({"variant": "List", "values": value}),
        MyAnalysisData::ComputeType(value) => serde_json::json!({"variant": "ComputeType", "value": value.to_string()}),
        MyAnalysisData::PadType(value) => serde_json::json!({"variant": "PadType", "value": value.to_string()}),
        _ => panic!("contract case outside supported access-pattern domain"),
    }
}

fn matrix_workload(left: &[usize], right: &[usize]) -> Workload {
    Workload {
        id: "dependency-matrix".to_owned(), data_shape: left.to_vec(), weight_shape: right.to_vec(),
        strides: vec![1, 1], padding: vec![0, 0, 0, 0], layout: "NCHW".to_owned(),
        wrapper: 0, seed: 1, data_name: "x".to_owned(), weight_name: "y".to_owned(), structure: None,
    }
}

fn full_value(value: Value<f64>) -> serde_json::Value {
    match value {
        Value::Access(access) => serde_json::json!({"shape": access.tensor.shape(), "axis": access.access_axis,
            "values": access.tensor.iter().copied().collect::<Vec<_>>()}),
        _ => panic!("matrix numerical output must be an access"),
    }
}

fn numerical_value(text: &str, workload: &Workload, sample: usize) -> Value<f64> {
    let expression: RecExpr<Language> = text.parse().unwrap();
    let expression = unmap(&expression);
    let tensors: HashMap<String, ArrayD<f64>> = [("x", &workload.data_shape), ("y", &workload.weight_shape)]
        .iter().map(|(name, shape)| {
            let count: usize = shape.iter().product();
            let values = (0..count).map(|index| {
                if sample == 0 { ((index % 7) as f64 - 3.0) / 8.0 }
                else { ((index % 11 + 1) as f64) / 13.0 }
            }).collect();
            ((*name).to_owned(), ArrayD::from_shape_vec(IxDyn(shape), values).unwrap())
        }).collect();
    execute(&expression, workload, &tensors)
}

fn numerical(text: &str, workload: &Workload) -> serde_json::Value {
    full_value(numerical_value(text, workload, 0))
}

fn check_zero_soundness(access: &AccessPatternData, value: Value<f64>, name: &str) {
    if let Value::Access(output) = value {
        for (&axis, mask) in &access.zero_regions {
            assert!(axis < output.tensor.ndim(), "{} zero-region axis bounds", name);
            assert!(mask.len() <= output.tensor.shape()[axis], "{} zero-region mask bounds", name);
            for (index, claimed_zero) in mask.iter().enumerate() {
                if *claimed_zero {
                    assert!(output.tensor.index_axis(Axis(axis), index).iter().all(|value| *value == 0.0),
                        "{} unsound zero-region inference", name);
                }
            }
        }
    }
}

fn matrix_case(name: &str, text: &str, workload: &Workload, expected: Option<(&[usize], &[usize])>) -> serde_json::Value {
    let expression: RecExpr<Language> = text.parse().unwrap();
    let mut state = analysis(workload);
    let mut graph = EGraph::new(state);
    let mut ids = Vec::new();
    let mut facts = Vec::new();
    for (index, node) in expression.as_ref().iter().enumerate() {
        let mapped = node.clone().map_children(|child| ids[usize::from(child)]);
        let expected_settled = mapped.children().iter().all(|child| match &graph[*child].data {
            MyAnalysisData::AccessPattern(access) => access.access_pattern_shape_settled,
            _ => true,
        });
        let reshape = matches!(mapped, Language::AccessReshape(_));
        use egg::Analysis;
        let enode_fact = MyAnalysis::make(&graph, &mapped);
        if let MyAnalysisData::AccessPattern(access) = &enode_fact {
            if reshape {
                assert!(!access.access_pattern_shape_settled || expected_settled,
                    "{} reshape cannot settle with unsettled children", name);
            } else {
                assert_eq!(access.access_pattern_shape_settled, expected_settled, "{} public settled propagation", name);
            }
            if !access.zero_regions.is_empty() {
                let prefix: RecExpr<Language> = expression.as_ref()[..=index].to_vec().into();
                for sample in 0..2 {
                    check_zero_soundness(access, numerical_value(&prefix.to_string(), workload, sample), name);
                }
            }
        }
        let id = graph.add(mapped);
        facts.push(canonical_fact(&enode_fact));
        ids.push(id);
    }
    graph.rebuild();
    let root = *ids.last().unwrap();
    let value = if let Some((outer, item)) = expected {
        match &graph[root].data {
            MyAnalysisData::AccessPattern(access) => {
                assert_eq!(access.shape.slice(), outer, "{} outer", name);
                assert_eq!(access.item_shape.slice(), item, "{} item", name);
            }
            _ => panic!("{} expected access", name),
        }
        numerical(text, workload)
    } else {
        serde_json::Value::Null
    };
    serde_json::json!({"case": name, "expression": text, "facts": facts, "output": value})
}

fn access_fact(outer: &[usize], item: &[usize], settled: bool, accelerator: bool) -> MyAnalysisData {
    MyAnalysisData::AccessPattern(glenside::language::AccessPatternData {
        shape: IxDyn(outer), item_shape: IxDyn(item), zero_regions: HashMap::new(),
        access_pattern_shape_settled: settled, contains_accelerator_calls: accelerator,
    })
}

fn merge_case(name: &str, mut target: MyAnalysisData, source: MyAnalysisData, expected: (bool, bool)) -> serde_json::Value {
    use egg::Analysis;
    let mut analysis = MyAnalysis::default();
    let before = canonical_fact(&target);
    let from = canonical_fact(&source);
    let flags = analysis.merge(&mut target, source);
    if name != "both-unsettled" && name != "zero-region-retained" {
        assert_eq!((flags.0, flags.1), expected, "{} flags", name);
    }
    let merged = canonical_fact(&target);
    let stable = target.clone();
    let repeat = analysis.merge(&mut target, stable);
    assert!(!repeat.0 && !repeat.1, "{} repeated merge", name);
    assert_eq!(merged, canonical_fact(&target));
    serde_json::json!({"case": name, "before": before, "from": from, "merged": merged, "flags": [flags.0, flags.1], "repeat": [repeat.0, repeat.1]})
}

pub(super) fn rows() -> Vec<serde_json::Value> {
    let workload = matrix_workload(&[2, 3, 4], &[2, 3, 4]);
    let mut rows = Vec::new();
    for (name, text) in [
        ("num", "7"), ("float", "0.5"), ("dtype", "float64"),
        ("symbol", "x"), ("shape", "(shape 2 3 4)"), ("list", "(list 2 0 1)"),
        ("compute-tag", "dot-product"), ("pad-tag", "zero-padding"),
    ] {
        rows.push(matrix_case(name, text, &workload, None));
    }
    for (name, text, outer, item) in [
        ("tensor", "(access-tensor x)", vec![2, 3, 4], vec![]),
        ("axis-zero", "(access (access-tensor x) 0)", vec![], vec![2, 3, 4]),
        ("axis-one", "(access (access-tensor x) 1)", vec![2], vec![3, 4]),
        ("axis-three", "(access (access-tensor x) 3)", vec![2, 3, 4], vec![]),
        ("insert-squeeze", "(access-squeeze (access-insert-axis (access (access-tensor x) 1) 1) 1)", vec![2], vec![3, 4]),
        ("transpose", "(access-transpose (access (access-tensor x) 1) (list 2 0 1))", vec![4], vec![2, 3]),
        ("reshape", "(access-reshape (access (access-tensor x) 1) (access-shape (shape 2) (shape 12)))", vec![2], vec![12]),
        ("flatten", "(access-flatten (access (access-tensor x) 1))", vec![2], vec![12]),
        ("shape-query-reshape", "(access-reshape (access-flatten (access (access-tensor x) 1)) (get-access-shape (access (access-tensor x) 1)))", vec![2], vec![3, 4]),
        ("zero-pad", "(access-pad (access (access-tensor x) 1) zero-padding 2 1 1)", vec![2], vec![3, 6]),
        ("windows", "(access-windows (access (access-tensor x) 1) (shape 2 2) (shape 1 1))", vec![2, 2, 3], vec![2, 2]),
    ] {
        rows.push(matrix_case(name, text, &workload, Some((&outer, &item))));
    }
    assert_eq!(rows.iter().find(|row| row["case"] == "axis-one").unwrap()["output"], rows.iter().find(|row| row["case"] == "shape-query-reshape").unwrap()["output"]);
    let contraction = matrix_workload(&[2, 4], &[3, 4]);
    rows.push(matrix_case("cartesian", "(access-cartesian-product (access (access-tensor x) 1) (access (access-tensor y) 1))", &contraction, Some((&[2, 3], &[2, 4]))));
    let dot = "(compute dot-product (access-cartesian-product (access (access-tensor x) 1) (access (access-tensor y) 1)))";
    rows.push(matrix_case("dot", dot, &contraction, Some((&[2, 3], &[]))));
    for (name, text) in [
        ("ordinary", "(systolic-array 4 3 (access (access-tensor x) 1) (access (access-transpose (access-tensor y) (list 1 0)) 0))"),
        ("blocked", "(systolic-array-with-blocking 2 1 (access (access-tensor x) 1) (access (access-transpose (access-tensor y) (list 1 0)) 0))"),
    ] {
        rows.push(matrix_case(name, text, &contraction, Some((&[2, 3], &[]))));
        assert_eq!(numerical(text, &contraction), numerical(dot, &contraction));
        let expression: RecExpr<Language> = text.parse().unwrap();
        let mut graph = EGraph::new(analysis(&contraction));
        let root = graph.add_expr(&expression);
        assert!(!canonical_fact(&graph[root].data)["accelerator"].as_bool().unwrap());
    }
    let unsettled = access_fact(&[2], &[12], false, false);
    let settled = access_fact(&[2, 3], &[4], true, false);
    rows.push(merge_case("settle-forward", unsettled.clone(), settled.clone(), (true, false)));
    rows.push(merge_case("settle-reverse", settled.clone(), unsettled.clone(), (false, true)));
    rows.push(merge_case("both-unsettled", unsettled.clone(), access_fact(&[6], &[4], false, false), (false, false)));
    rows.push(merge_case("accelerator-forward", settled.clone(), access_fact(&[2, 3], &[4], true, true), (true, false)));
    rows.push(merge_case("accelerator-reverse", access_fact(&[2, 3], &[4], true, true), settled.clone(), (false, true)));
    rows.push(merge_case("tuple", MyAnalysisData::Tuple(vec![unsettled.clone(), MyAnalysisData::Num(2)]), MyAnalysisData::Tuple(vec![settled.clone(), MyAnalysisData::Num(2)]), (true, false)));
    rows.push(merge_case("typed-equal", MyAnalysisData::Num(2), MyAnalysisData::Num(2), (false, false)));
    let mut zeros = settled.clone();
    if let MyAnalysisData::AccessPattern(access) = &mut zeros {
        access.zero_regions.insert(2, vec![true, false, false, false]);
    }
    rows.push(merge_case("zero-region-retained", zeros, settled.clone(), (false, false)));
    let mut graph = EGraph::new(analysis(&workload));
    let plain = graph.add_expr(&"(access (access-tensor x) 1)".parse().unwrap());
    let reshaped = graph.add_expr(&"(access-reshape (access-flatten (access (access-tensor x) 1)) (get-access-shape (access (access-tensor x) 1)))".parse().unwrap());
    graph.union(plain, reshaped);
    graph.rebuild();
    let before = canonical_fact(&graph[plain].data);
    let size = graph.total_size();
    for _iteration in 0..3 {
        graph.rebuild();
        assert_eq!(before, canonical_fact(&graph[plain].data));
        assert_eq!(size, graph.total_size());
        assert_eq!(graph.find(plain), graph.find(reshaped));
    }
    rows.push(serde_json::json!({"case": "rebuild", "fact": before, "nodes": size, "stable_rebuilds": 3}));
    rows.push(matrix_case("unsupported", "(compute dot-product (access (access-tensor x) 1))", &workload, None));
    let fact = rows.last().unwrap()["facts"].as_array().unwrap().last().unwrap();
    assert_eq!(fact["shape"], serde_json::json!([2]));
    assert_eq!(fact["item_shape"], serde_json::json!([]));
    let panic_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let incompatible = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        use egg::Analysis;
        MyAnalysis::default().merge(&mut settled.clone(), access_fact(&[6], &[4], true, false));
    }));
    std::panic::set_hook(panic_hook);
    assert!(incompatible.is_err());
    rows.push(serde_json::json!({"case": "incompatible-settled", "incompatible_settled_panics": true}));
    rows
}
