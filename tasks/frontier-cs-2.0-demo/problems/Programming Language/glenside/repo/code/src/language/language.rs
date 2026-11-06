use crate::language::RelayOperator::*;
use egg::{define_language, DidMerge, EGraph, Id, Language as LanguageTrait};
use itertools::{any, multizip};
use log::debug;
use ndarray::Ix0;
use ndarray::{s, Dimension, Ix, IxDyn};
use ordered_float::NotNan;

use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::convert::TryInto;
use std::fmt::Display;
use std::iter::FromIterator;
use std::str::FromStr;

define_language! {
    pub enum Language {
        // (systolic-array <rows (usize)> <cols (usize)> <access-0> <access-1>)
        // Represents a systolic array of size rows X cols, fed with two
        // accesses.
        // This is Scott's weight-stationary systolic array design. It reads in
        // two matrices: first matrix is in the layout [M, N] and second is in
        // the layout [N, O]. The systolic array computes the matrix
        // multiplication, leading to a matrix with layout [M, O].
        // The systolic array expects exactly one shape for the second argument:
        // [N, O]. These correspond directly to the rows/cols parameters of the
        // systolic array. The first argument is partially constrained: its
        // second dimension must be N, but its first dimension may be any
        // length.
        // In terms of Glenside accesses, we expect <access-0> to have shape [M]
        // [N], and <access-1> to have shape [] [N, O].
        // TODO(@gussmith23) How to make the M argument "programmable"?
        // TODO(@gussmith23) do we need to specify rows and cols? You can infer these
        // from the size of the input, but it's also useful for searching.
        "systolic-array" = SystolicArray([Id; 4]),

        // Same as the systolic array above, but relies on Scott's blocking code
        // instead of relying on Glenside to discover the blocking. By
        // "blocking", we mean splitting up a matrix multiply to run on a
        // smaller systolic array.
        "systolic-array-with-blocking" = SystolicArrayWithBlocking([Id; 4]),

        // (systolic-array-conv2d-nchw-oihw-with-blocking
        //  <rows: Num> <cols: Num>
        //  <weights: Access> <data: Access>
        //  <kh: Num> <kw: Num>
        //  <stride-h: Num> <stride-w: Num>)
        // A systolic array operating in conv2d mode, with data in layout NCHW
        // and weights in layout OIHW. We don't actually have an atom for this,
        // but it's currently used as an intermediate to help discover
        // systolic-array-conv2d-nhwc-hwio-with-blocking.
        "systolic-array-conv2d-nchw-oihw-with-blocking" = SystolicArrayConv2dNchwOihwWithBlocking([Id; 8]),

        // (systolic-array-conv2d-nhwc-hwio-with-blocking
        //  <rows: Num> <cols: Num>
        //  <weights: Access> <data: Access>
        //  <kh: Num> <kw: Num>
        //  <stride-h: Num> <stride-w: Num>)
        // A systolic array operating in conv2d mode, with data in layout NHWC
        // and weights in layout HWIO. Scott should have an atom for this at
        // some point.
        "systolic-array-conv2d-nhwc-hwio-with-blocking" = SystolicArrayConv2dNhwcHwioWithBlocking([Id; 8]),

        // (systolic-array-conv2d-im2col-nchw-oihw-with-blocking <rows: Num>
        // <cols: Num> <weights: Access> <data: Access> <kh: Num> <kw:
        // Num> <stride-h: Num> <stride-w: Num>)
        // A systolic array operating in conv2d-im2col mode, with data in layout
        // NCHW and weights in layout OIHW. We don't actually have an atom for
        // this, but it's currently used as an intermediate to help discover
        // systolic-array-conv2d-im2col-nhwc-hwio-with-blocking. Conv2d im2col
        // mode indicates that the systolic array is reading in the image data
        // and transforming it to im2col on the fly.
        "systolic-array-conv2d-im2col-nchw-oihw-with-blocking" = SystolicArrayConv2dIm2colNchwOihwWithBlocking([Id; 8]),
        "systolic-array-conv2d-im2col-nhwc-hwio-with-blocking" = SystolicArrayConv2dIm2colNhwcHwioWithBlocking([Id; 8]),

        // (access-windows <access> <filters-shape: Shape> <stride-shape: Shape>)
        // Form the windows which will be convolved over.
        // TODO(@gussmith23) AccessWindows shouldn't be specific to filters.
        // AccessWindows is used in other contexts too, i.e. pooling.
        "access-windows" = AccessWindows([Id; 3]),

        // (shape-of <tensor>)
        // Returns the shape of the tensor.
        // TODO(@gussmith) Choose between ([Id; 1]) and (Id) and be consistent
        // When describing the arguments of a construct that takes a single Id
        // argument (like shape-of), we can use (Id) or ([Id; 1]). I'm not sure
        // which is better, but I should choose one and be consistent.
        "shape-of" = ShapeOf([Id; 1]),

        // (slice-shape <shape> <dim>)
        // Slices a shape by taking dimensions >= <dim>.
        "slice-shape" = SliceShape([Id; 2]),

        // (shape-insert-axis <shape: Shape> <axis: usize>)
        // Inserts an axis with value 1.
        "shape-insert-axis" = ShapeInsertAxis([Id; 2]),

        // (shape-remove-axis <shape: Shape> <axis: usize>)
        // Removes axis from shape.
        "shape-remove-axis" = ShapeRemoveAxis([Id; 2]),

        // (access <tensor> <dim>)
        // The most basic access pattern.
        // Let <tensor> have dims d0, .., dn.
        // Interprets <tensor> as a shaped list of shape d0, .., d(<dim>-1)
        // whose elements are of shape d<dim>, .., dn.
        "access" = Access([Id; 2]),

        // (access-transpose <a: access> <new-order: list>)
        // Uses numpy.transpose() semantics. Reorders axes in an access.
        // Does not change the access dimension.
        "access-transpose" = AccessTranspose([Id; 2]),

        // (access-cartesian-product <access1> <access2>)
        // Cartesian product access pattern.
        // Assume <access1> has shape
        // [a1, ..., an]
        // and <access2> has shape
        // [b1, ..., bm].
        // Both must have the same item shape,
        // [c1, ..., co]
        // Outputs a tensor of shape
        // [a1, ..., an, b1, ..., bm, 2, c1, ..., co]
        // which represents the cartesian product of the items in both accesses.
        "access-cartesian-product" = AccessCartesianProduct([Id; 2]),

        // (compute <compute-type> <access>)
        // Compute over the items in <access>.
        //
        // Compute types:
        //
        // dot-product
        // Expects an item shape of
        // [n, a0, ..., am]
        // Where n specifies the tuple multiplicity and [a0, ..., am] is the
        // shape of the tensors to be dot-producted with one another.
        "compute" = Compute([Id; 2]),

        // (get-access-shape <access>)
        // Returns the shape of the access.
        "get-access-shape" = GetAccessShape([Id;1]),
        // The below comment is wrong:
        //
        // This shouldn't actually be needed at the moment. We are handling all
        // statically-sized networks, and so anywhere where we would have used
        // this, we should be able to just plug in a literal access-shape. If
        // and when we start supporting dynamic networks, this will become
        // needed.
        //
        // Turns out we need this for flatten-unflatten. We can't just use
        // static shapes because the underlying shape might change as the shapes
        // "settle". It would be better to not have this "settling" at all, but
        // for now, we do.

        // (access-reshape <access> <shape>)
        // Reshapes the access to have the given
        "access-reshape" = AccessReshape([Id; 2]),

        // (access-flatten <access>)
        // Flattens the access's shape and item shape.
        "access-flatten" = AccessFlatten(Id),

        // (shape <usize>...)
        // Shape literal.
        "shape" = Shape(Box<[Id]>),

        // (list <usize>...)
        // List literal
        "list" = List(Box<[Id]>),

        // (construct-tuple <access> <access> ...)
        // Tuple Construction
        "construct-tuple" = ConstructTuple(Box<[Id]>),

        // (tuple-get-item <tuple> <i>)
        // Get the item at the ith index of tuple
        "tuple-get-item" = TupleGetItem([Id;2]),

        // (access-shape <shape: shape> <item-shape: shape>)
        // Access shape literal.
        "access-shape" = AccessShape([Id;2]),

        // (access-slice <access> <axis (usize)> <low (usize)> <high (usize)>)
        // Slices into <access> at axis <axis>, slicing the half-open range
        // [<low>, <high>).
        // TODO(@gussmith23) Implement access-slice-item
        // If axis >= access.shape.ndim(), it slices into access.item_shape.
        // This is me being lazy and not wanting to implement separate
        // access-slice-shape and access-slice-item operators for right now.
        "access-slice" = AccessSlice([Id; 4]),

        // (access-concatenate <a0> <a1> <axis (usize)>)
        // Concatenate accesses <a0> and <a1> along <axis>.
        // TODO(@gussmith23) Implement access-concatenate-item
        // If axis >= access.shape.ndim(), it concatenates along dimensions in
        // access.item_shape.
        "access-concatenate" = AccessConcatenate([Id; 3]),

        // (access-pair <a0> <a1>)
        // Simply pair every item of a0 with every item of a1.
        "access-pair" = AccessPair([Id; 2]),

        // (access-shift-right <a0>)
        // Shifts a dimension from shape to item shape.
        "access-shift-right" = AccessShiftRight(Id),

        // (access-tensor <t>)
        // Access a tensor literal.
        "access-tensor" = AccessTensor(Id),

        // (access-pad <a>
        //             <pad-type (PadType)>
        //             <axis (usize)> <pad-before (usize)> <pad-after (usize)>)
        // Pads a tensor at the given axis.
        "access-pad" = AccessPad([Id; 5]),

        // (access-squeeze <a> <axis (usize)>)
        "access-squeeze" = AccessSqueeze([Id; 2]),

        // (access-insert-axis <a> <axis (usize)>)
        "access-insert-axis" = AccessInsertAxis([Id; 2]),

        // (access-broadcast <a> <shape: shape>)
        // Simple broadcasting. <a> and <shape> must have the same total number
        // of dimensions. All dimensions in <a> must either match the
        // corresponding dimension in <shape> or be 1.
        "access-broadcast" = AccessBroadcast([Id; 2]),

        // (access-literal <literal: Literal>)
        // Access a literal. This may be able to be folded in to some other
        // access pattern, later on. It fits in with access-tensor as a "access
        // pattern constructor"; it takes something that isn't an access pattern
        // and converts it to an access pattern.
        "access-literal" = AccessLiteral(Id),

        // (literal <val: Float64>)
        // A literal value. Can only represent 0-dimensional values for now, but
        // in the future, we can and should support array constants.
        "literal" = Literal(Id),

        // (relay-operator-call <relay-operator: RelayOperator> <args>...)
        "relay-operator-call" = RelayOperatorCall(Box<[Id]>),

        "accelerator-call" = AcceleratorCall(Box<[Id]>),

        // (constant-tensor <value> <shape>)
        "constant-tensor" = ConstantTensor([Id; 2]),

        Num(i64),

        DataType(DataType),

        // Important that this go after usize, so that usizes are parsed as
        // usizes, not as floats.
        NotNanFloat64(NotNan<f64>),

        RelayOperator(RelayOperator),
        RelayActivationLayout(RelayActivationLayout),
        RelayKernelLayout(RelayKernelLayout),

        // pad-type: zero-padding
        // (No other options right now)
        PadType(PadType),

        ComputeType(ComputeType),

        AcceleratorFunc(AcceleratorFunc),

        Symbol(String),
    }
}

// TODO(@gussmith23) We need to just make a full-fledged Relay dialect.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RelayOperator {
    /// (relay-operator relay-batch-norm-inference <data: access>
    ///  <gamma: access> <beta: access>
    ///  <moving_mean: access> <moving_var: access>
    ///  <axis: usize> <epsilon: float>)
    /// The batch-norm-at-inference-time operator. We don't currently support
    /// normal batch norm, which is a training-time operator, but we do support
    /// its inference-time simplified version.
    /// TODO(@gussmith23) How to handle batch norms?
    RelayBatchNormInference,

    /// (relay-operator relay-softmax <data: access> <axis: int>)
    RelaySoftmax,

    /// (relay-operator relay-relu <data: access>)
    RelayReLU,

    /// (relay-operator relay-leaky-relu <data: access> <alpha: Float64>)
    RelayLeakyReLU,

    /// (relay-operator relay-max-pool2d <data: access>
    ///  <pool size: shape> <strides: shape> <padding: shape>
    ///  <layout: RelayActivationLayout>)
    RelayMaxPool2D,

    /// (relay-operator relay-global-avg-pool2d <data: access>
    ///  <layout: RelayActivationLayout>)
    RelayGlobalAvgPool2D,

    /// (relay-operator relay-avg-pool2d <data: access>
    /// <pool_size: shape> <strides: shape> <padding: shape> <layout: RelayActivationLayout>
    /// )
    RelayAvgPool2D,

    /// (relay-operator relay-upsampling <data: access> <scale_h: Float64> <scale_w: Float64>
    /// <layout: RelayActivationLayout>)
    RelayUpSampling,

    /// (relay-operator relay-batch-flatten <data: access>)
    RelayBatchFlatten,

    /// (relay-operator relay-bias-add <data: access> <bias: access>
    ///  <axis: usize>)
    RelayBiasAdd,

    /// (relay-operator relay-dense <data: access> <weight: access>)
    RelayDense,

    /// (relay-operator relay-reshape <data: access> <shape: shape>)
    RelayReshape,

    /// (relay-operator-call relay-conv1d <data: access> <kernel: access>
    ///  <strides: shape> <padding: shape>)
    RelayConv1D,

    /// (relay-operator relay-erf <data: access>)
    RelayErf,

    /// (relay-operator relay-mean <data: access> <axis: usize>)
    RelayMean,

    /// (relay-operator relay-add <a: access> <b: access>)
    RelayAdd,

    /// (relay-operator relay-multiply <a: access> <b: access>)
    RelayMultiply,

    /// (relay-operator relay-divide <a: access> <b: access>)
    RelayDivide,

    /// (relay-operator relay-sigmoid <data: access>)
    RelaySigmoid,

    /// (relay-operator relay-maximum <a: access> <b: access>)
    RelayMaximum,

    /// (relay-operator relay-minimum <a:access> <b: access>)
    RelayMinimum,

    /// (relay-operator relay-conv2d <data: access> <kernel: access>
    ///  <strides: shape> <padding: shape> <group: usize>
    ///  <channels: usize> <kernel_size: shape> <activation_layout>
    ///  <kernel_layout>)
    RelayConv2D,

    /// (relay-operator relay-split <data: access> <indices_or_sections: usize> <axis: usize>)
    RelaySplit,

    /// (relay-operator relay-cast <data: access> <dtype: DataType>)
    RelayCast,

    /// (relay-operator relay-clip <data: access> <a_min: NonNanf64> <a_max: NonNanf64>)
    RelayClip,

    /// (relay-operator relay-left-shift <data: access> <nbits: i32>)
    RelayLeftShift,

    /// (relay-operator relay-right-shift <data: access> <nbits: i32>)
    RelayRightShift,

    /// (relay-operator round <data: access>)
    RelayRound,

    /// (relay-operator relay-take <data: access> <indices: access> <axis: usize>)
    RelayTake,

    /// (relay-operator relay-dropout <data: access> <rate: f64>)
    RelayDropout,

    /// (relay-operator relay-tanh <data: access>)
    RelayTanh,

    /// (relay-operator relay-stack <data: access> ... <axis: i32>)
    RelayStack,

    /// (relay-operator relay-log-softmax <data: access> <axis: usize>)
    RelayLogSoftmax,

    /// (relay-operator relay-strided-slice <data: access> <begin: shape> <end: shape> <strides: shape>)
    RelayStridedSlice,

    RelayLayerNorm,

    RelayBatchMatmul,

    RelayZeros,

    RelaySqrt,

    RelayNegative,

    /// (relay-operator-call relay-expand-dims <data: access> <axis: int> <num_newaxis: int>)
    RelayExpandDims,

    /// (relay-operator-call relay-pad <data: access>
    ///                                <pad_width: shape>)
    /// Note that the pad_width is a flattened version of the list of tuples
    /// which is used in Relay.
    /// Padding value is assumed to be zero.
    RelayPad,

    /// (relay-operator-call relay-concatenate <to-concat: Tuple of access> <axis: num>)
    RelayConcatenate,

    /// (relay-operator-call relay-transpose <data: access> <axes: list of num>)
    RelayTranspose,

    /// (relay-operator-call relay-squeeze <data: access> <axes: list of num>)
    RelaySqueeze,
    RelayCopy,
}

/// All variants of [`RelayOperator`].
/// TODO(@gussmith23) We shouldn't always need this. Currently just used for
/// from_relay.
pub static RELAY_OPS: &[RelayOperator] = &[
    RelayBatchNormInference,
    RelaySoftmax,
    RelayReLU,
    RelayLeakyReLU,
    RelayMaxPool2D,
    RelayGlobalAvgPool2D,
    RelayAvgPool2D,
    RelayUpSampling,
    RelayBatchFlatten,
    RelayBiasAdd,
    RelayDense,
    RelayReshape,
    RelayConv1D,
    RelayErf,
    RelayMean,
    RelayAdd,
    RelayMultiply,
    RelayDivide,
    RelaySigmoid,
    RelayMaximum,
    RelayMinimum,
    RelayConv2D,
    RelaySplit,
    RelayCast,
    RelayClip,
    RelayLeftShift,
    RelayRightShift,
    RelayRound,
    RelayTake,
    RelayDropout,
    RelayTanh,
    RelayStack,
    RelayLogSoftmax,
    RelayStridedSlice,
    RelayLayerNorm,
    RelayBatchMatmul,
    RelayZeros,
    RelaySqrt,
    RelayNegative,
    RelayExpandDims,
    RelayPad,
    RelayConcatenate,
    RelayTranspose,
    RelaySqueeze,
    RelayCopy,
];

impl FromStr for RelayOperator {
    type Err = ();
    fn from_str(input: &str) -> Result<RelayOperator, Self::Err> {
        match input {
            "relay-squeeze" => Ok(RelayOperator::RelaySqueeze),
            "relay-transpose" => Ok(RelayOperator::RelayTranspose),
            "relay-concatenate" => Ok(RelayOperator::RelayConcatenate),
            "relay-batch-norm-inference" => Ok(RelayOperator::RelayBatchNormInference),
            "relay-softmax" => Ok(RelayOperator::RelaySoftmax),
            "relay-relu" => Ok(RelayOperator::RelayReLU),
            "relay-max-pool2d" => Ok(RelayOperator::RelayMaxPool2D),
            "relay-global-avg-pool2d" => Ok(RelayOperator::RelayGlobalAvgPool2D),
            "relay-batch-flatten" => Ok(RelayOperator::RelayBatchFlatten),
            "relay-bias-add" => Ok(RelayOperator::RelayBiasAdd),
            "relay-add" => Ok(RelayOperator::RelayAdd),
            "relay-sigmoid" => Ok(RelayOperator::RelaySigmoid),
            "relay-avg-pool2d" => Ok(RelayOperator::RelayAvgPool2D),
            "relay-upsampling" => Ok(RelayOperator::RelayUpSampling),
            "relay-maximum" => Ok(RelayOperator::RelayMaximum),
            "relay-minimum" => Ok(RelayOperator::RelayMinimum),
            "relay-leaky-relu" => Ok(RelayOperator::RelayLeakyReLU),
            "relay-dense" => Ok(RelayOperator::RelayDense),
            "relay-reshape" => Ok(RelayOperator::RelayReshape),
            "relay-conv1d" => Ok(RelayOperator::RelayConv1D),
            "relay-erf" => Ok(RelayOperator::RelayErf),
            "relay-mean" => Ok(RelayOperator::RelayMean),
            "relay-multiply" => Ok(RelayOperator::RelayMultiply),
            "relay-conv2d" => Ok(RelayOperator::RelayConv2D),
            "relay-split" => Ok(RelayOperator::RelaySplit),
            "relay-cast" => Ok(RelayOperator::RelayCast),
            "relay-clip" => Ok(RelayOperator::RelayClip),
            "relay-left-shift" => Ok(RelayOperator::RelayLeftShift),
            "relay-right-shift" => Ok(RelayOperator::RelayRightShift),
            "relay-round" => Ok(RelayOperator::RelayRound),
            "relay-take" => Ok(RelayOperator::RelayTake),
            "relay-dropout" => Ok(RelayOperator::RelayDropout),
            "relay-stack" => Ok(RelayOperator::RelayStack),
            "relay-log-softmax" => Ok(RelayOperator::RelayLogSoftmax),
            "relay-strided-slice" => Ok(RelayOperator::RelayStridedSlice),
            "relay-layer-norm" => Ok(RelayOperator::RelayLayerNorm),
            "relay-batch-matmul" => Ok(RelayOperator::RelayBatchMatmul),
            "relay-zeros" => Ok(RelayOperator::RelayZeros),
            "relay-negative" => Ok(RelayOperator::RelayNegative),
            "relay-sqrt" => Ok(RelayOperator::RelaySqrt),
            "relay-expand-dims" => Ok(RelayOperator::RelayExpandDims),
            "relay-pad" => Ok(RelayOperator::RelayPad),
            "relay-divide" => Ok(RelayOperator::RelayDivide),
            _ => Err(()),
        }
    }
}
impl Display for RelayOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                RelayOperator::RelaySqueeze => "relay-squeeze",
                RelayOperator::RelayTranspose => "relay-transpose",
                RelayOperator::RelayConcatenate => "relay-concatenate",
                RelayOperator::RelayStridedSlice => "relay-strided-slice",
                RelayOperator::RelayBatchNormInference => "relay-batch-norm-inference",
                RelayOperator::RelaySoftmax => "relay-softmax",
                RelayOperator::RelayReLU => "relay-relu",
                RelayOperator::RelayLeakyReLU => "relay-leaky-relu",
                RelayOperator::RelayMaxPool2D => "relay-max-pool2d",
                RelayOperator::RelayGlobalAvgPool2D => "relay-global-avg-pool2d",
                RelayOperator::RelayBatchFlatten => "relay-batch-flatten",
                RelayOperator::RelayBiasAdd => "relay-bias-add",
                RelayOperator::RelayAdd => "relay-add",
                RelayOperator::RelaySigmoid => "relay-sigmoid",
                RelayOperator::RelayAvgPool2D => "relay-avg-pool2d",
                RelayOperator::RelayUpSampling => "relay-upsampling",
                RelayOperator::RelayMaximum => "relay-maximum",
                RelayOperator::RelayMinimum => "relay-minimum",
                RelayOperator::RelayDense => "relay-dense",
                RelayOperator::RelayReshape => "relay-reshape",
                RelayOperator::RelayConv1D => "relay-conv1d",
                RelayOperator::RelayErf => "relay-erf",
                RelayOperator::RelayMean => "relay-mean",
                RelayOperator::RelayMultiply => "relay-mul",
                RelayOperator::RelayConv2D => "relay-conv2d",
                RelayOperator::RelaySplit => "relay-split",
                RelayOperator::RelayCast => "relay-cast",
                RelayOperator::RelayClip => "relay-clip",
                RelayOperator::RelayLeftShift => "relay-left-shift",
                RelayOperator::RelayRightShift => "relay-right-shift",
                RelayOperator::RelayRound => "relay-round",
                RelayOperator::RelayTake => "relay-take",
                RelayOperator::RelayDropout => "relay-dropout",
                RelayOperator::RelayTanh => "relay-tanh",
                RelayOperator::RelayStack => "relay-stack",
                RelayOperator::RelayLogSoftmax => "relay-log-softmax",
                RelayOperator::RelayLayerNorm => "relay-layer-norm",
                RelayOperator::RelayBatchMatmul => "relay-batch-matmul",
                RelayOperator::RelayZeros => "relay-zeros",
                RelayOperator::RelaySqrt => "relay-sqrt",
                RelayOperator::RelayNegative => "relay-negative",
                RelayOperator::RelayExpandDims => "relay-expand-dims",
                RelayOperator::RelayPad => "relay-pad",
                RelayOperator::RelayDivide => "relay-divide",
                RelayOperator::RelayCopy => "relay-copy",
            }
        )
    }
}

/// Only for use in [`RelayOperatorCall`]s.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RelayActivationLayout {
    NCHW,
    NHWC,
}
impl FromStr for RelayActivationLayout {
    type Err = ();
    fn from_str(input: &str) -> Result<RelayActivationLayout, Self::Err> {
        match input {
            "relay-activation-layout-nhwc" => Ok(RelayActivationLayout::NHWC),
            "relay-activation-layout-nchw" => Ok(RelayActivationLayout::NCHW),
            _ => Err(()),
        }
    }
}
impl Display for RelayActivationLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                RelayActivationLayout::NHWC => "relay-activation-layout-nhwc",
                RelayActivationLayout::NCHW => "relay-activation-layout-nchw",
            }
        )
    }
}

/// Only for use in [`RelayOperatorCall`]s.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RelayKernelLayout {
    OIHW,
    HWIO,
}
impl FromStr for RelayKernelLayout {
    type Err = ();
    fn from_str(input: &str) -> Result<RelayKernelLayout, Self::Err> {
        match input {
            "relay-kernel-layout-hwio" => Ok(RelayKernelLayout::HWIO),
            "relay-kernel-layout-oihw" => Ok(RelayKernelLayout::OIHW),
            _ => Err(()),
        }
    }
}
impl Display for RelayKernelLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                RelayKernelLayout::HWIO => "relay-kernel-layout-hwio",
                RelayKernelLayout::OIHW => "relay-kernel-layout-oihw",
            }
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComputeType {
    DotProduct,
    ReduceSum,
    ReLU,
    Sqrt,
    Negative,
    /// Expects item shape of `a x b1 x .. x bn`. Performs an elementwise
    /// addition of the `a` tensors of size `b1 x .. x bn`.
    /// TODO(@gussmith) Multiple-arg compute feels clunky and ad-hoc.
    /// Should figure out an explicit way to define access multiple-stream
    /// access patterns.
    ElementwiseAdd,
    /// Expects item shape of `a x b1 x .. x bn`. Performs an elementwise
    /// multiplication of the `a` tensors of size `b1 x .. x bn`.
    ElementwiseMul,
    ElementwiseDiv,
    /// Takes the max across all elements in each item. Reduces any item shape
    /// to a scalar.
    ReduceMax,
    /// Computes softmax. Currently expects access axis to be 0. Unsure how to
    /// define softmax for other access patterns.
    Softmax,
    /// For an item shape of `a1 x a2 x ...`, returns an item shape of `1` where
    /// the returned scalar is the mean of the `a1 x a2 x ...`-shaped tensor.
    ReduceMean,
}
impl FromStr for ComputeType {
    type Err = ();
    fn from_str(input: &str) -> Result<ComputeType, Self::Err> {
        match input {
            "dot-product" => Ok(ComputeType::DotProduct),
            "reduce-sum" => Ok(ComputeType::ReduceSum),
            "reduce-max" => Ok(ComputeType::ReduceMax),
            "relu" => Ok(ComputeType::ReLU),
            "sqrt" => Ok(ComputeType::Sqrt),
            "negative" => Ok(ComputeType::Negative),
            "elementwise-add" => Ok(ComputeType::ElementwiseAdd),
            "elementwise-mul" => Ok(ComputeType::ElementwiseMul),
            "elementwise-div" => Ok(ComputeType::ElementwiseDiv),
            "softmax" => Ok(ComputeType::Softmax),
            "reduce-mean" => Ok(ComputeType::ReduceMean),
            _ => Err(()),
        }
    }
}
impl Display for ComputeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ComputeType::DotProduct => "dot-product",
                ComputeType::ReduceSum => "reduce-sum",
                ComputeType::ReduceMax => "reduce-max",
                ComputeType::ReLU => "relu",
                ComputeType::Sqrt => "sqrt",
                ComputeType::Negative => "negative",
                ComputeType::ElementwiseAdd => "elementwise-add",
                ComputeType::ElementwiseMul => "elementwise-mul",
                ComputeType::ElementwiseDiv => "elementwise-div",
                ComputeType::Softmax => "softmax",
                ComputeType::ReduceMean => "reduce-mean",
            }
        )
    }
}

/// Specifies how to pick the values we pad with.
#[derive(Debug, PartialEq, Eq, Hash, Clone, PartialOrd, Ord, Copy)]
pub enum PadType {
    /// Pad with zeroes.
    ZeroPadding,
    /// Pad with minimum representable number in the number system.
    MinPadding,
}
impl FromStr for PadType {
    type Err = ();
    fn from_str(input: &str) -> Result<PadType, Self::Err> {
        match input {
            "zero-padding" => Ok(PadType::ZeroPadding),
            "min-padding" => Ok(PadType::MinPadding),
            _ => Err(()),
        }
    }
}
impl Display for PadType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                PadType::ZeroPadding => "zero-padding",
                PadType::MinPadding => "min-padding",
            }
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AcceleratorFunc {
    FlexLinear,
    FlexLSTM,
    VTADense,
    VTAConv1D,
    HlsCNNConv2D,
    // (accelerator-call flex-maxpool <access>)
    //
    // Compute's FlexASR's maxpool operator. The input access should be of
    // shape ((),(t, h)) where t is the number of timesteps and h is the
    // number of hidden states. t should be divisible by 2; h should be
    // divisible by 16. The result is an access pattern with shape ((),(t/2,
    // h)).
    //
    // TODO(@gussmith23) Add tests for flexasr-maxpool.
    FlexASRMaxPool,
}

impl FromStr for AcceleratorFunc {
    type Err = ();

    fn from_str(s: &str) -> Result<AcceleratorFunc, Self::Err> {
        match s {
            "flex-linear" => Ok(AcceleratorFunc::FlexLinear),
            "flex-lstm" => Ok(AcceleratorFunc::FlexLSTM),
            "vta-dense" => Ok(AcceleratorFunc::VTADense),
            "vta-conv1d" => Ok(AcceleratorFunc::VTAConv1D),
            "hlscnn-conv2d" => Ok(AcceleratorFunc::HlsCNNConv2D),
            "flex-maxpool" => Ok(AcceleratorFunc::FlexASRMaxPool),
            _ => Err(()),
        }
    }
}

impl Display for AcceleratorFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                AcceleratorFunc::FlexLinear => "flex-linear",
                AcceleratorFunc::FlexLSTM => "flex-lstm",
                AcceleratorFunc::VTADense => "vta-dense",
                AcceleratorFunc::VTAConv1D => "vta-conv1d",
                AcceleratorFunc::HlsCNNConv2D => "hlscnn-conv2d",
                AcceleratorFunc::FlexASRMaxPool => "flex-maxpool",
            }
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AcceleratorFuncData {
    pattern: AcceleratorFunc,
    accelerator: String,
}

// TODO(@gussmith23) Pick a better analysis name.
#[derive(Debug, Clone, PartialEq)]
pub enum MyAnalysisData {
    Literal(ndarray::ArrayD<f64>),
    Num(i64),
    DataType(DataType),
    AccessPattern(AccessPatternData),
    Shape(ShapeData),
    Tuple(Vec<MyAnalysisData>),
    // TODO(@gussmith23) Needed?
    //Tensor(TensorData),
    ComputeType(ComputeType),
    PadType(PadType),
    List(Vec<usize>),
    RelayOperator(RelayOperator),
    RelayActivationLayout(RelayActivationLayout),
    RelayKernelLayout(RelayKernelLayout),
    AcceleratorFunc(AcceleratorFuncData),
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, PartialOrd, Ord, Copy)]
pub enum DataType {
    Bool,
    Int(usize),
    Float(usize),
    Uint(usize),
}

impl Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                DataType::Bool => "bool".into(),
                DataType::Int(x) => format!("int{}", x),
                DataType::Float(x) => format!("float{}", x),
                DataType::Uint(x) => format!("uint{}", x),
            }
        )
    }
}

impl FromStr for DataType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (dtype, bits) = match s.find(char::is_numeric) {
            Some(idx) => s.split_at(idx),
            None => (s, "32"),
        };
        if dtype == "bool" {
            return Ok(DataType::Bool);
        }
        if let Ok(bits) = bits.parse::<usize>() {
            match dtype {
                "int" => Ok(DataType::Int(bits)),
                "float" => Ok(DataType::Float(bits)),
                "uint" => Ok(DataType::Uint(bits)),
                _ => Err(format!("Not supported: {}", dtype)),
            }
        } else {
            Err(format!("cannot parse bits"))
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShapeData {
    pub shape: IxDyn,
    pub dtype: DataType,
}

/// New version of rangeset.
pub trait RangeSet2 {
    type Index;

    /// Inserts elements, shifting existing ranges as needed.
    fn insert_elements(&mut self, index: Self::Index, num_elements_inserted: usize);

    /// Updates ranges as if `num_elements_removed` elements were removed at
    /// `index`.
    fn remove_elements(&mut self, index: Self::Index, num_elements_removed: usize);

    /// Checks whether `range` is covered by the ranges in this set.
    fn covered(&self, range: (Self::Index, Self::Index)) -> bool;

    /// Adds range. Ranges are half-open.
    fn add_range(&mut self, range: (Self::Index, Self::Index));
}

type BoolVecRangeSet = Vec<bool>;
impl RangeSet2 for BoolVecRangeSet {
    type Index = usize;

    fn insert_elements(&mut self, index: Self::Index, num_elements_inserted: usize) {
        // Make index-1 the last valid index.
        if index >= self.len() {
            self.resize(index, false);
        }
        *self = self[..index]
            .iter()
            .chain(std::iter::repeat(&false).take(num_elements_inserted))
            .chain(self[index..].iter())
            .cloned()
            .collect();
    }

    fn remove_elements(&mut self, index: Self::Index, num_elements_removed: usize) {
        *self = self[..index]
            .iter()
            .chain(self[index + num_elements_removed..].iter())
            .cloned()
            .collect()
    }

    fn covered(&self, range: (Self::Index, Self::Index)) -> bool {
        // If the top end of the range is not actually represented, then those
        // values are implicitly false and so the range is not covered.
        // Otherwise, check that the values are all true.
        range.1 <= self.len() && self[range.0..range.1].iter().all(|v| *v)
    }

    fn add_range(&mut self, range: (Self::Index, Self::Index)) {
        // Make range.1-1 the last valid index.
        if range.1 > self.len() {
            self.resize(range.1, false);
        }
        for i in range.0..range.1 {
            self[i] = true;
        }
    }
}

#[cfg(test)]
mod bool_vec_range_set_tests {
    use super::*;

    #[test]
    fn insert_elements_0() {
        let mut range_set = BoolVecRangeSet::default();
        range_set.add_range((0, 3));
        range_set.add_range((2, 6));
        range_set.add_range((4, 8));
        range_set.add_range((7, 10));
        range_set.insert_elements(5, 5);
        assert!(range_set.covered((0, 3)));
        assert!(range_set.covered((2, 5)));
        assert!(range_set.covered((10, 11)));
        assert!(range_set.covered((4, 5)));
        assert!(range_set.covered((10, 13)));
        assert!(range_set.covered((12, 15)));
    }

    #[test]
    fn insert_elements_1() {
        let mut range_set = BoolVecRangeSet::default();
        range_set.add_range((0, 3));
        range_set.add_range((2, 6));
        range_set.add_range((4, 8));
        range_set.add_range((7, 10));
        range_set.insert_elements(5, 5);
        range_set.add_range((5, 10));
        assert!(range_set.covered((0, 3)));
        assert!(range_set.covered((2, 11)));
        assert!(range_set.covered((4, 13)));
        assert!(range_set.covered((12, 15)));
    }

    #[test]
    fn remove_elements() {
        let mut range_set = BoolVecRangeSet::default();
        range_set.add_range((0, 3));
        range_set.add_range((2, 6));
        range_set.add_range((5, 8));
        range_set.add_range((9, 12));
        range_set.add_range((10, 14));
        range_set.remove_elements(5, 5);
        assert!(range_set.covered((0, 3)));
        assert!(range_set.covered((2, 5)));
        assert!(range_set.covered((5, 7)));
        assert!(range_set.covered((5, 9)));
    }

    #[test]
    fn covered() {
        let mut range_set = BoolVecRangeSet::default();
        range_set.add_range((0, 3));
        range_set.add_range((5, 6));
        range_set.add_range((6, 8));
        range_set.add_range((10, 12));
        range_set.add_range((11, 14));
        assert!(range_set.covered((0, 2)));
        assert!(!range_set.covered((0, 4)));
        assert!(!range_set.covered((2, 5)));
        assert!(!range_set.covered((3, 5)));
        assert!(range_set.covered((5, 7)));
        assert!(range_set.covered((5, 8)));
        assert!(!range_set.covered((5, 9)));
        assert!(range_set.covered((10, 14)));
        assert!(!range_set.covered((10, 16)));
        assert!(!range_set.covered((22, 23)));
    }

    #[test]
    fn test() {
        let mut range_set = BoolVecRangeSet::default();
        range_set.insert_elements(0, 1);
        range_set.add_range((0, 1));
        range_set.insert_elements(33, 2);
        range_set.add_range((33, 35));
        assert!(range_set.covered((0, 1)));
        assert!(!range_set.covered((1, 33)));
        assert!(range_set.covered((33, 35)));
    }
}

/// Used to represent ranges over a set from 0..n, for some n. Ranges are
/// half-open.
type RangeHashSet = HashSet<(usize, usize)>;
pub enum RangeInsertStrategy {
    /// If elements are inserted in the middle of a range, the range gets split
    /// in two.
    BreakRanges,
    /// If elements are inserted in the middle of a range, they get folded into
    /// the range.
    PreserveRanges,
}
pub trait RangeSet {
    type Index;

    /// Updates ranges as if `num_elements_inserted` elements were inserted at
    /// `index`, according to `strategy`.
    fn insert_elements(
        &mut self,
        strategy: RangeInsertStrategy,
        index: Self::Index,
        num_elements_inserted: usize,
    );

    /// Updates ranges as if `num_elements_removed` elements were removed at
    /// `index`.
    fn remove_elements(&mut self, index: Self::Index, num_elements_removed: usize);

    /// Checks whether `range` is covered by the ranges in this set.
    fn covered(&self, range: (Self::Index, Self::Index)) -> bool;

    /// Adds range. Ranges are half-open.
    fn add_range(&mut self, range: (Self::Index, Self::Index));
}
impl RangeSet for RangeHashSet {
    type Index = usize;

    fn insert_elements(
        &mut self,
        strategy: RangeInsertStrategy,
        index: usize,
        num_elements_inserted: usize,
    ) {
        let mut new_ranges = Vec::default();
        for (low, high) in self.drain() {
            assert!(low <= high);
            match strategy {
                RangeInsertStrategy::PreserveRanges => {
                    let (new_low, new_high) = if index < low {
                        (low + num_elements_inserted, high + num_elements_inserted)
                    } else if index >= low && index <= high {
                        (low, high + num_elements_inserted)
                    } else if index > high {
                        (low, high)
                    } else {
                        unreachable!()
                    };
                    new_ranges.push((new_low, new_high));
                }
                RangeInsertStrategy::BreakRanges => {
                    match {
                        if index <= low {
                            (
                                Some((low + num_elements_inserted, high + num_elements_inserted)),
                                None,
                            )
                        } else if index > low && index < high {
                            (
                                Some((low, index)),
                                Some((index + num_elements_inserted, high + num_elements_inserted)),
                            )
                        } else if index >= high {
                            (Some((low, high)), None)
                        } else {
                            unreachable!()
                        }
                    } {
                        (Some(range1), Some(range2)) => {
                            new_ranges.push(range1);
                            new_ranges.push(range2);
                        }
                        (Some(range1), None) => {
                            new_ranges.push(range1);
                        }
                        _ => panic!(),
                    };
                }
            }
        }

        for range in new_ranges.iter() {
            self.insert(*range);
        }
    }

    fn remove_elements(&mut self, index: usize, num_elements_removed: usize) {
        let new_ranges = self
            .drain()
            .filter_map(|(low, high): (usize, usize)| {
                let new_low = if low <= index {
                    low
                } else if low > index {
                    low - std::cmp::min(num_elements_removed, low - index)
                } else {
                    unreachable!()
                };
                let new_high = if index >= high {
                    high
                } else if index < high {
                    high - std::cmp::min(num_elements_removed, high - index)
                } else {
                    unreachable!()
                };

                // If the range is valid and nonempty
                if new_low < new_high {
                    Some((new_low, new_high))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        for new_range in new_ranges {
            self.insert(new_range);
        }
    }

    /// I'm hoping this implementation will be fast.
    fn covered(&self, range: (Self::Index, Self::Index)) -> bool {
        let mut to_be_covered =
            HashSet::<_, std::collections::hash_map::RandomState>::from_iter(range.0..range.1);

        for (low, high) in self.iter() {
            to_be_covered = to_be_covered
                .difference(&HashSet::from_iter(*low..*high))
                .cloned()
                .collect();
        }

        to_be_covered.is_empty()
    }

    /// Adds range. Ranges are half-open.
    fn add_range(&mut self, range: (Self::Index, Self::Index)) {
        self.insert(range);
    }
}

#[cfg(test)]
mod range_hash_set_tests {
    use super::*;

    #[test]
    fn insert_elements_break_ranges() {
        let mut range_set = RangeHashSet::default();
        range_set.add_range((0, 3));
        range_set.add_range((2, 6));
        range_set.add_range((4, 8));
        range_set.add_range((7, 10));
        range_set.insert_elements(RangeInsertStrategy::BreakRanges, 5, 5);
        assert_eq!(range_set.len(), 6);
        assert!(range_set.contains(&(0, 3)));
        assert!(range_set.contains(&(2, 5)));
        assert!(range_set.contains(&(10, 11)));
        assert!(range_set.contains(&(4, 5)));
        assert!(range_set.contains(&(10, 13)));
        assert!(range_set.contains(&(12, 15)));
    }

    #[test]
    fn insert_elements_preserve_ranges() {
        let mut range_set = RangeHashSet::default();
        range_set.add_range((0, 3));
        range_set.add_range((2, 6));
        range_set.add_range((4, 8));
        range_set.add_range((7, 10));
        range_set.insert_elements(RangeInsertStrategy::PreserveRanges, 5, 5);
        assert_eq!(range_set.len(), 4);
        assert!(range_set.contains(&(0, 3)));
        assert!(range_set.contains(&(2, 11)));
        assert!(range_set.contains(&(4, 13)));
        assert!(range_set.contains(&(12, 15)));
    }

    #[test]
    fn remove_elements() {
        let mut range_set = RangeHashSet::default();
        range_set.add_range((0, 3));
        range_set.add_range((2, 6));
        range_set.add_range((5, 8));
        range_set.add_range((9, 12));
        range_set.add_range((10, 14));
        range_set.remove_elements(5, 5);
        assert_eq!(range_set.len(), 4);
        assert!(range_set.contains(&(0, 3)));
        assert!(range_set.contains(&(2, 5)));
        assert!(range_set.contains(&(5, 7)));
        assert!(range_set.contains(&(5, 9)));
    }

    #[test]
    fn covered() {
        let mut range_set = RangeHashSet::default();
        range_set.add_range((0, 3));
        range_set.add_range((5, 6));
        range_set.add_range((6, 8));
        range_set.add_range((10, 12));
        range_set.add_range((11, 14));
        assert!(range_set.covered((0, 2)));
        assert!(!range_set.covered((0, 4)));
        assert!(!range_set.covered((2, 5)));
        assert!(!range_set.covered((3, 5)));
        assert!(range_set.covered((5, 7)));
        assert!(range_set.covered((5, 8)));
        assert!(!range_set.covered((5, 9)));
        assert!(range_set.covered((10, 14)));
        assert!(!range_set.covered((10, 16)));
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AccessPatternData {
    pub shape: IxDyn,
    pub item_shape: IxDyn,
    /// Regions proven to be zero-valued. The outermost map maps a axis index to
    /// a set of usize tuples, indicating the half-open indices [low, high)
    /// which are known to be zero in that axis.
    /// TODO(@gussmith23) Might want to replace this with full partial eval
    /// I realized while implementing this that I could just implement partial
    /// eval, and it would do the same for me. I don't think it would be more
    /// time efficient, though, and I'm even more certain that it wouldn't be
    /// space efficient.
    pub zero_regions: HashMap<Ix, BoolVecRangeSet>,
    /// Indicates whether the access pattern shape for this eclass has
    /// "settled". An access pattern shape for an eclass containing only Relay
    /// nodes is likely to change, because Relay nodes treat access pattern
    /// shapes e.g. ((a, b), (c, d)) as tensor shapes e.g. (a, b, c, d), and
    /// often the Relay nodes choose either the shape or the item shape to store
    /// the tensor shape. The access pattern is thus likely to change when the
    /// class gets merged, so we call it "unsettled". An enode is considered to
    /// have a settled shape if and only if:
    ///
    /// - The node is a Glenside node, and
    /// - All of the node's children of access pattern type are also settled.
    ///
    /// A value-preserving access-reshape may conservatively remain unsettled.
    /// Both this representation and child-propagated settling are supported.
    ///
    /// (This specifies how we implement make().)
    ///
    /// An eclass's shape is settled if at least one of its nodes is settled.
    /// (This specifies how we implement merge().)
    ///
    pub access_pattern_shape_settled: bool,
    pub contains_accelerator_calls: bool,
}

impl AccessPatternData {
    /// Convenience method for getting the access pattern dimensions as a
    /// vector.
    /// ```
    /// assert_eq!(
    ///     glenside::language::AccessPatternData {
    ///         shape: ndarray::IxDyn(&[1, 2, 3]),
    ///         item_shape: ndarray::IxDyn(&[4, 5]),
    ///         zero_regions: std::collections::HashMap::default(),
    ///         access_pattern_shape_settled: false,
    ///         contains_accelerator_calls: false,
    ///     }
    ///     .as_vec(),
    ///     vec![1, 2, 3, 4, 5]
    /// );
    /// ```
    pub fn as_vec(&self) -> Vec<usize> {
        self.shape
            .slice()
            .iter()
            .chain(self.item_shape.slice().iter())
            .cloned()
            .collect::<Vec<_>>()
    }
}

impl std::ops::Index<usize> for AccessPatternData {
    type Output = ndarray::Ix;
    fn index(&self, index: usize) -> &Self::Output {
        if index < self.shape.ndim() {
            &self.shape[index]
        } else {
            &self.item_shape[index - self.shape.ndim()]
        }
    }
}

impl std::ops::IndexMut<usize> for AccessPatternData {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index < self.shape.ndim() {
            &mut self.shape[index]
        } else {
            &mut self.item_shape[index - self.shape.ndim()]
        }
    }
}

pub fn access_windows_resulting_shape(
    access_shape: &IxDyn,
    filters_shape: &IxDyn,
    stride_shape: &IxDyn,
) -> Vec<usize> {
    assert_eq!(access_shape.ndim(), stride_shape.ndim());
    assert_eq!(filters_shape.ndim(), stride_shape.ndim());

    multizip((
        access_shape.slice().iter(),
        filters_shape.slice().iter(),
        stride_shape.slice().iter(),
    ))
    .map(
        |(&dim_len, &kernel_dim_len, &stride): (&usize, &usize, &usize)| {
            let total_dim_len = dim_len;
            assert!(
                total_dim_len >= kernel_dim_len,
                "{} !>= {}",
                total_dim_len,
                kernel_dim_len
            );
            let num_spots = total_dim_len - (kernel_dim_len - 1);
            (num_spots + stride - 1) / stride
        },
    )
    .collect()
}

// #[derive(Debug, Clone, PartialEq)]
// pub struct TensorData {
//     shape: IxDyn,
// }

// TODO(@gussmith23) Pick a better analysis name.
#[derive(Debug, Clone, PartialEq)]
pub struct MyAnalysisDataLegacyData {
    pub(crate) shape: Option<IxDyn>,
    pub(crate) usize_value: Option<usize>,
}
#[derive(Default)]
pub struct MyAnalysis {
    pub name_to_shape: HashMap<String, Vec<usize>>,
    pub name_to_dtype: HashMap<String, DataType>,
}
impl MyAnalysis {
    /// Legacy function: gets Num value as a usize. Before Num, we instead had a
    /// Num construct.
    pub fn get_usize(id: Id, egraph: &EGraph<Language, MyAnalysis>) -> usize {
        match &egraph[id].data {
            &MyAnalysisData::Num(s) => s.try_into().unwrap(),
            _ => panic!(),
        }
    }
    pub(crate) fn get_shape(id: Id, egraph: &EGraph<Language, MyAnalysis>) -> &IxDyn {
        match &egraph[id].data {
            MyAnalysisData::Shape(s) => &s.shape,
            _ => panic!(),
        }
    }
    pub(crate) fn get_dtype(id: Id, egraph: &EGraph<Language, MyAnalysis>) -> &DataType {
        match &egraph[id].data {
            MyAnalysisData::Shape(s) => &s.dtype,
            _ => panic!(),
        }
    }
    pub(crate) fn get_shape_of_value(id: Id, egraph: &EGraph<Language, MyAnalysis>) -> &IxDyn {
        match &egraph[id].data {
            MyAnalysisData::Shape(s) => &s.shape,
            _ => panic!(),
        }
    }
}

include!("solver_extensions/access_analysis.rs");
