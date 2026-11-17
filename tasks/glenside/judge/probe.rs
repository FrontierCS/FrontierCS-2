use egg::{Applier, CostFunction, EGraph, Extractor, Id, Language as EggLanguage, PatternAst, RecExpr, Rewrite, Runner, SearchMatches, Subst, Symbol};
use glenside::language::from_relay::{access_transpose, conv2d};
use glenside::language::interpreter::{interpret, Value};
use glenside::language::{ComputeType, Language, MyAnalysis, MyAnalysisData};
use ndarray::{ArrayD, Dimension, IxDyn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, Read};
use std::time::Duration;
use std::sync::Arc;
mod fixed_mapping;
mod analysis_contract;

#[derive(Clone, Deserialize)]
struct TensorSpec {
    name: String,
    shape: Vec<usize>,
    access_axis: usize,
}

#[derive(Clone, Deserialize)]
struct OuterConcatSpec {
    concat_axis: usize,
    left: Vec<TensorSpec>,
    right: Vec<TensorSpec>,
}

#[derive(Clone, Deserialize)]
struct Workload {
    id: String,
    data_shape: Vec<usize>,
    weight_shape: Vec<usize>,
    strides: Vec<usize>,
    padding: Vec<usize>,
    layout: String,
    wrapper: usize,
    seed: u64,
    data_name: String,
    weight_name: String,
    #[serde(default)]
    structure: Option<OuterConcatSpec>,
}

#[derive(Deserialize)]
struct Request {
    mode: String,
    workloads: Vec<Workload>,
    witnesses: Option<Vec<Option<String>>>,
}

#[derive(Serialize)]
struct Measurement {
    workload: String,
    rewrite_time_seconds: f64,
    egraph_node_growth: usize,
    witnesses: Vec<Option<String>>,
}

fn analysis(workload: &Workload) -> MyAnalysis {
    let mut shapes = HashMap::new();
    shapes.insert(workload.data_name.clone(), workload.data_shape.clone());
    shapes.insert(workload.weight_name.clone(), workload.weight_shape.clone());
    if let Some(structure) = &workload.structure {
        for tensor in structure.left.iter().chain(structure.right.iter()) {
            shapes.insert(tensor.name.clone(), tensor.shape.clone());
        }
    }
    MyAnalysis { name_to_shape: shapes, name_to_dtype: HashMap::new() }
}

fn access_from_spec(expression: &mut RecExpr<Language>, spec: &TensorSpec) -> Id {
    let symbol = expression.add(Language::Symbol(spec.name.clone()));
    let tensor = expression.add(Language::AccessTensor(symbol));
    let axis = expression.add(Language::Num(spec.access_axis as i64));
    expression.add(Language::Access([tensor, axis]))
}

fn explicit_outer_concat(expression: &mut RecExpr<Language>, structure: &OuterConcatSpec) -> Id {
    assert_eq!(structure.left.len(), 2);
    assert_eq!(structure.right.len(), 2);
    let left_a0 = access_from_spec(expression, &structure.left[0]);
    let right_b0 = access_from_spec(expression, &structure.right[0]);
    let left = expression.add(Language::AccessCartesianProduct([
        left_a0,
        right_b0,
    ]));
    let left_a1 = access_from_spec(expression, &structure.left[1]);
    let right_b1 = access_from_spec(expression, &structure.right[1]);
    let right = expression.add(Language::AccessCartesianProduct([
        left_a1,
        right_b1,
    ]));
    let axis = expression.add(Language::Num(structure.concat_axis as i64));
    let concatenated = expression.add(Language::AccessConcatenate([left, right, axis]));
    let compute_type = expression.add(Language::ComputeType(ComputeType::DotProduct));
    expression.add(Language::Compute([compute_type, concatenated]))
}

fn original(workload: &Workload) -> RecExpr<Language> {
    if let Some(structure) = &workload.structure {
        let mut expression = RecExpr::default();
        explicit_outer_concat(&mut expression, structure);
        return expression.to_string().parse().unwrap();
    }
    let mut expression: RecExpr<Language> = format!("(access-tensor {})", workload.data_name).parse().unwrap();
    let data_id = Id::from(expression.as_ref().len() - 1);
    let weight_symbol = expression.add(Language::Symbol(workload.weight_name.clone()));
    let weight_id = expression.add(Language::AccessTensor(weight_symbol));
    let root = conv2d(
        &mut expression, data_id, &workload.data_shape, weight_id,
        &workload.weight_shape, &workload.strides, &workload.padding,
        &[1, 1], 1, &workload.layout,
        if workload.layout == "NHWC" { "HWIO" } else { "OIHW" }, "", false,
    );
    match workload.wrapper {
        1 => { access_transpose(&mut expression, root, &[0, 3, 1, 2]); }
        2 => {
            let axis = expression.add(Language::Num(0));
            let expanded = expression.add(Language::AccessInsertAxis([root, axis]));
            expression.add(Language::AccessSqueeze([expanded, axis]));
        }
        3 => {
            let first = access_transpose(&mut expression, root, &[2, 0, 3, 1]);
            access_transpose(&mut expression, first, &[1, 3, 0, 2]);
        }
        _ => {}
    }
    expression.to_string().parse().unwrap()
}

struct MappedCost;
impl CostFunction<Language> for MappedCost {
    type Cost = usize;
    fn cost<Child>(&mut self, node: &Language, mut costs: Child) -> usize
    where Child: FnMut(Id) -> usize {
        let own: usize = match node {
            Language::Compute(_) => 1_000_000,
            _ => 1,
        };
        node.fold(own, |total, child| total.saturating_add(costs(child)))
    }
}

fn accelerator(node: &Language) -> bool {
    matches!(node, Language::SystolicArray(_) | Language::SystolicArrayWithBlocking(_)
        | Language::SystolicArrayConv2dNchwOihwWithBlocking(_)
        | Language::SystolicArrayConv2dNhwcHwioWithBlocking(_)
        | Language::SystolicArrayConv2dIm2colNchwOihwWithBlocking(_)
        | Language::SystolicArrayConv2dIm2colNhwcHwioWithBlocking(_)
        | Language::AcceleratorCall(_) | Language::RelayOperatorCall(_))
}

struct GenericApplier(Arc<dyn Applier<Language, MyAnalysis> + Send + Sync>);
impl Applier<Language, MyAnalysis> for GenericApplier {
    fn apply_matches(&self, graph: &mut EGraph<Language, MyAnalysis>, matches: &[SearchMatches<Language>], name: Symbol) -> Vec<Id> {
        let before: Vec<Language> = graph.classes().flat_map(|class| class.nodes.iter()).filter(|node| accelerator(node)).cloned().collect();
        let result = self.0.apply_matches(graph, matches, name);
        let permitted: std::collections::HashSet<Language> = before.into_iter().map(|node| node.map_children(|child| graph.find(child))).collect();
        assert!(graph.classes().flat_map(|class| class.nodes.iter()).filter(|node| accelerator(node)).all(|node| permitted.contains(&node.clone().map_children(|child| graph.find(child)))));
        result
    }
    fn apply_one(&self, graph: &mut EGraph<Language, MyAnalysis>, class: Id, substitution: &Subst, ast: Option<&PatternAst<Language>>, name: Symbol) -> Vec<Id> {
        self.0.apply_one(graph, class, substitution, ast, name)
    }
}

fn generic_rules() -> Vec<Rewrite<Language, MyAnalysis>> {
    let mut rules = vec![
        glenside::language::rewrites::flatten_unflatten_any_access(),
        glenside::language::rewrites::bubble_reshape_through_cartesian_product(),
        glenside::language::rewrites::bubble_reshape_through_compute_dot_product(),
        glenside::language::rewrites::bubble_access_concatenate_through_compute_dot_product_not_item_axis(),
    ];
    for rule in &mut rules { rule.applier = Arc::new(GenericApplier(rule.applier.clone())); }
    rules
}

fn trusted_mapping_rules() -> Vec<Rewrite<Language, MyAnalysis>> {
    vec![fixed_mapping::systolic_array(), fixed_mapping::systolic_array_with_blocking(1, 1)]
}

fn measure(workload: &Workload) -> Measurement {
    let expression = original(workload);
    let mut times = Vec::new();
    let mut growth = Vec::new();
    let mut witnesses: Vec<Option<String>> = Vec::new();
    for _ in 0..3 {
        let mut graph = EGraph::new(analysis(workload));
        let root = graph.add_expr(&expression);
        graph.rebuild();
        let initial = graph.total_number_of_nodes();
        let mut rules = generic_rules();
        rules.extend(trusted_mapping_rules());
        let runner = Runner::<_, _, ()>::new(MyAnalysis::default())
            .with_egraph(graph).with_iter_limit(30).with_node_limit(100_000)
            .with_time_limit(Duration::from_secs(10)).run(&rules);
        let report = runner.report();
        times.push(report.total_time);
        let current_growth = report.egraph_nodes.checked_sub(initial).unwrap();
        growth.push(current_growth);
        let extracted = Extractor::new(&runner.egraph, MappedCost).find_best(root).1;
        let mapped = extracted.as_ref().iter().filter(|node| matches!(node, Language::SystolicArray(_) | Language::SystolicArrayWithBlocking(_))).count();
        let current = if mapped >= 1 && extracted.as_ref().len() < 4096 {
            Some(extracted.to_string())
        } else { None };
        assert_eq!(runner.egraph.find(runner.egraph.lookup_expr(&expression).unwrap()), runner.egraph.find(root));
        witnesses.push(current);
    }
    summarize_repetitions(workload.id.clone(), times, growth, witnesses)
}

fn summarize_repetitions(workload: String, mut times: Vec<f64>, growth: Vec<usize>, witnesses: Vec<Option<String>>) -> Measurement {
    assert_eq!(times.len(), 3);
    assert_eq!(growth.len(), 3);
    assert_eq!(witnesses.len(), 3);
    times.sort_by(|left, right| left.partial_cmp(right).unwrap());
    // A time-bounded search may stop at different sizes. Charge its largest graph;
    // the trusted verifier still requires all three witnesses for discovery.
    Measurement { workload, rewrite_time_seconds: times[1], egraph_node_growth: *growth.iter().max().unwrap(), witnesses }
}

#[cfg(test)]
mod repetition_tests {
    use super::summarize_repetitions;

    #[test]
    fn variable_searches_preserve_partial_witnesses_and_worst_growth() {
        let row = summarize_repetitions("variable".into(), vec![3.0, 1.0, 2.0],
            vec![20, 40, 30], vec![Some("mapped".into()), None, Some("mapped".into())]);
        assert_eq!(row.rewrite_time_seconds, 2.0);
        assert_eq!(row.egraph_node_growth, 40);
        assert_eq!(row.witnesses.len(), 3);
        assert!(row.witnesses[1].is_none());
    }
}

fn unmap(expression: &RecExpr<Language>) -> RecExpr<Language> {
    let mut result = RecExpr::default();
    let mut ids: Vec<Id> = Vec::new();
    for node in expression.as_ref() {
        let mapped = node.clone().map_children(|child| ids[usize::from(child)]);
        let id = match mapped {
            Language::SystolicArray([_, _, left, right]) | Language::SystolicArrayWithBlocking([_, _, left, right]) => {
                let transposed = access_transpose(&mut result, right, &[1, 0]);
                let axis = result.add(Language::Num(1));
                let original_right = result.add(Language::Access([transposed, axis]));
                let product = result.add(Language::AccessCartesianProduct([left, original_right]));
                let operation = result.add(Language::ComputeType(ComputeType::DotProduct));
                result.add(Language::Compute([operation, product]))
            }
            _ => result.add(mapped),
        };
        ids.push(id);
    }
    result.to_string().parse().unwrap()
}

fn shape_node(expression: &mut RecExpr<Language>, dims: &[usize]) -> Id {
    let ids: Vec<Id> = dims.iter().map(|dimension| expression.add(Language::Num(*dimension as i64))).collect();
    expression.add(Language::Shape(ids.into_boxed_slice()))
}

fn reify(value: &Value<f64>, expression: &mut RecExpr<Language>, tensors: &mut HashMap<String, ArrayD<f64>>) -> Id {
    match value {
        Value::Access(access) => {
            let name = format!("value-{}", tensors.len());
            tensors.insert(name.clone(), access.tensor.as_standard_layout().to_owned());
            let symbol = expression.add(Language::Symbol(name));
            let tensor = expression.add(Language::AccessTensor(symbol));
            let axis = expression.add(Language::Num(access.access_axis as i64));
            expression.add(Language::Access([tensor, axis]))
        }
        Value::Tensor(tensor) => {
            let name = format!("value-{}", tensors.len());
            tensors.insert(name.clone(), tensor.as_standard_layout().to_owned());
            expression.add(Language::Symbol(name))
        }
        Value::Num(number) => expression.add(Language::Num(*number as i64)),
        Value::Shape(shape) => shape_node(expression, shape.slice()),
        Value::List(values) => {
            let ids: Vec<Id> = values.iter().map(|number| expression.add(Language::Num(*number as i64))).collect();
            expression.add(Language::List(ids.into_boxed_slice()))
        }
        Value::AccessShape(shape, axis) => {
            let outer = shape_node(expression, &shape.slice()[..*axis]);
            let inner = shape_node(expression, &shape.slice()[*axis..]);
            expression.add(Language::AccessShape([outer, inner]))
        }
        Value::ComputeType(operation) => expression.add(Language::ComputeType(operation.clone())),
        Value::PadType(padding) => expression.add(Language::PadType(*padding)),
        _ => panic!("unsupported value"),
    }
}

fn execute(expression: &RecExpr<Language>, workload: &Workload, tensors: &HashMap<String, ArrayD<f64>>) -> Value<f64> {
    let mut graph = EGraph::new(analysis(workload));
    let mut ids = Vec::new();
    let mut values: Vec<Value<f64>> = Vec::new();
    for node in expression.as_ref() {
        let graph_node = node.clone().map_children(|child| ids[usize::from(child)]);
        let graph_id = graph.add(graph_node);
        ids.push(graph_id);
        let value = match node {
            Language::Symbol(name) => Value::Tensor(tensors.get(name).unwrap().clone()),
            Language::GetAccessShape([child]) => match &values[usize::from(*child)] {
                Value::Access(access) => Value::AccessShape(IxDyn(access.tensor.shape()), access.access_axis),
                _ => panic!("shape requires access"),
            },
            _ => {
                let mut local = RecExpr::default();
                let mut environment = HashMap::new();
                let local_node = node.clone().map_children(|child| reify(&values[usize::from(child)], &mut local, &mut environment));
                if matches!(local_node, Language::AccessFlatten(_) | Language::AccessTensor(_)) {
                    let shape = match &graph[graph_id].data { MyAnalysisData::AccessPattern(shape) => shape, _ => panic!() };
                    let outer = shape_node(&mut local, shape.shape.slice());
                    let inner = shape_node(&mut local, shape.item_shape.slice());
                    let access_shape = local.add(Language::AccessShape([outer, inner]));
                    let child = match local_node {
                        Language::AccessFlatten(child) => child,
                        _ => local.add(local_node),
                    };
                    local.add(Language::AccessReshape([child, access_shape]));
                } else {
                    local.add(local_node);
                }
                let borrowed = environment.iter().map(|(name, tensor)| (name.as_str(), tensor.clone())).collect();
                interpret::<f64>(&local, local.as_ref().len() - 1, &borrowed)
            }
        };
        if let Value::Access(access) = &value {
            assert!(access.tensor.len() <= 2_000_000);
            match &graph[graph_id].data {
                MyAnalysisData::AccessPattern(shape) => {
                    assert_eq!(access.tensor.shape(), shape.as_vec().as_slice());
                    assert_eq!(access.access_axis, shape.shape.ndim());
                }
                _ => panic!("access analysis required"),
            }
        }
        values.push(value);
    }
    values.pop().unwrap()
}

fn tensor(shape: &[usize], state: &mut u64) -> ArrayD<f64> {
    let count: usize = shape.iter().product();
    let values = (0..count).map(|_| {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        ((*state % 7) as i64 - 3) as f64 / 8.0
    }).collect();
    ArrayD::from_shape_vec(IxDyn(shape), values).unwrap()
}

fn verify(workload: &Workload, witness: &Option<String>) -> bool {
    let text = match witness { Some(text) if text.len() < 262144 => text, _ => return false };
    let mapped: RecExpr<Language> = text.parse().unwrap();
    assert!(mapped.as_ref().iter().any(|node| matches!(node, Language::SystolicArray(_) | Language::SystolicArrayWithBlocking(_))));
    let mut graph = EGraph::new(analysis(workload));
    let mapped_root = graph.add_expr(&mapped);
    let expected_expression = original(workload);
    let expected_root = graph.add_expr(&expected_expression);
    match (&graph[mapped_root].data, &graph[expected_root].data) {
        (MyAnalysisData::AccessPattern(left), MyAnalysisData::AccessPattern(right)) => {
            assert_eq!(left.shape, right.shape);
            assert_eq!(left.item_shape, right.item_shape);
        }
        _ => return false,
    }
    let before_mapping = unmap(&mapped);
    for sample in 0..2 {
        let mut state = workload.seed.wrapping_add(sample * 918271).max(1);
        let mut tensors = HashMap::new();
        tensors.insert(workload.data_name.clone(), tensor(&workload.data_shape, &mut state));
        tensors.insert(workload.weight_name.clone(), tensor(&workload.weight_shape, &mut state));
        if let Some(structure) = &workload.structure {
            for spec in structure.left.iter().chain(structure.right.iter()) {
                if !tensors.contains_key(&spec.name) {
                    tensors.insert(spec.name.clone(), tensor(&spec.shape, &mut state));
                }
            }
        }
        match (execute(&expected_expression, workload, &tensors), execute(&before_mapping, workload, &tensors)) {
            (Value::Access(expected), Value::Access(actual)) => {
                if expected.access_axis != actual.access_axis || expected.tensor != actual.tensor { return false; }
            }
            _ => return false,
        }
    }
    true
}

fn main() {
    let default_panic_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let request: Request = serde_json::from_str(&input).unwrap();
    if request.mode == "analysis-contract" {
        std::panic::set_hook(default_panic_hook);
        println!("{}", serde_json::to_string(&analysis_contract::rows()).unwrap());
    } else if request.mode == "registration-check" {
        println!("{}", analysis_contract::registration_check());
    } else if request.mode == "verify" {
        let witnesses = request.witnesses.unwrap();
        assert_eq!(request.workloads.len(), witnesses.len());
        let results: Vec<bool> = request.workloads.iter().zip(witnesses.iter()).map(|(workload, witness)| {
            std::panic::catch_unwind(|| verify(workload, witness)).unwrap_or(false)
        }).collect();
        println!("{}", serde_json::to_string(&results).unwrap());
    } else {
        let results: Vec<Measurement> = request.workloads.iter().map(measure).collect();
        println!("{}", serde_json::to_string(&results).unwrap());
    }
}
