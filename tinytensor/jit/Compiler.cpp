#include <tt/jit/Compiler.h>
#include <tt/jit/OpNode.h>
#include <tt/jit/Ops.h>
#include <tt/tensor.h>
#include <algorithm>
#include <iostream>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

namespace tinytensor::jit {

// This hides the actual MLIR objects from the header
struct JITCompiler::Impl {
    mlir::MLIRContext context;
    mlir::ModuleOp module;
    std::unique_ptr<mlir::OpBuilder> builder;

    Impl() {
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::tosa::TosaDialect>();
    }
};

JITCompiler::JITCompiler() : impl(std::make_unique<Impl>()) {}
JITCompiler::~JITCompiler() = default;
JITCompiler::JITCompiler(JITCompiler&&) noexcept = default;
JITCompiler& JITCompiler::operator=(JITCompiler&&) noexcept = default;
CompilerVisitor::CompilerVisitor(mlir::OpBuilder& b, mlir::ModuleOp& m, mlir::MLIRContext& c)
    : builder(b), module(m), context(c) {}

namespace {
Shape infer_shape(const std::shared_ptr<OpNode>& node) {

    return std::visit([&](const auto& op) -> Shape {
        using T = std::decay_t<decltype(op)>;
        if constexpr (std::is_same_v<T, InputOp>) {
            return op.shape;
        }
        else if constexpr (std::is_same_v<T, BroadcastOp>) {
            return op.target_shape;
        }
        else if constexpr (std::is_same_v<T, ReshapeOp>) {
            return op.target_shape;
        }
        else if constexpr (std::is_same_v<T, ReluOp>) {
            return infer_shape(node->inputs()[0]);
        }
        else if constexpr (std::is_same_v<T, AddOp>) {
            return infer_shape(node->inputs()[0]);
        }
        else if constexpr (std::is_same_v<T, MatMulOp>) {
            Shape lhs = infer_shape(node->inputs()[0]);
            Shape rhs = infer_shape(node->inputs()[1]);

            // if (lhs.ndim() != 3 || rhs.ndim() != 3) {
            //     TT_ERROR("SizeError: TOSA MatMul requires Rank-3 inputs");
            // }

            int B = std::max(lhs[0], rhs[0]);  // Broadcasting for now
            return {B, lhs[1], rhs[2]};
        }
        else {
            TT_ERROR("Unknown OpType in shape inference");
        }
    }, node->op());
}

void collect_inputs_recursive(const std::shared_ptr<OpNode>& node,
                              std::unordered_map<uintptr_t, bool>& visited,
                              std::vector<InputOp>& inputs) {
    auto key = reinterpret_cast<uintptr_t>(node.get());
    if (visited.count(key)) return;
    visited[key] = true;

    if (std::holds_alternative<InputOp>(node->op())) {
        inputs.push_back(std::get<InputOp>(node->op()));
    }

    for (const auto& in : node->inputs()) {
        collect_inputs_recursive(in, visited, inputs);
    }
}


std::vector<InputOp> get_sorted_graph_inputs(const std::shared_ptr<OpNode>& root) {
    std::unordered_map<uintptr_t, bool> visited;
    std::vector<InputOp> inputs;
    collect_inputs_recursive(root, visited, inputs);

    std::ranges::sort(inputs, [](const InputOp& a, const InputOp& b) {
        return a.id < b.id;
    });

    return inputs;
}

mlir::Type to_mlir_type(mlir::OpBuilder& builder, const Shape& shape, ScalarType dtype) {
    const std::vector<int>& dims = shape.to_vec();

    // DEBUG PRINT
    std::cout << "Debug to_mlir_type: ";
    for(auto d : dims) std::cout << d << " ";
    std::cout << std::endl;

    std::vector<int64_t> mlir_shape(dims.begin(), dims.end());
    mlir::Type element_type = builder.getF32Type();
    return mlir::RankedTensorType::get(mlir_shape, element_type);
}

}


mlir::Value CompilerVisitor::get_mlir_value(const std::shared_ptr<OpNode>& node) const {
    if (!node_value_map.contains(node)) {
        TT_ERROR("Compiler Error: Node dependency not found in SSA map.");
    }
    return node_value_map.at(node);
}

void CompilerVisitor::set_mlir_value(const OpType&, mlir::Value val) {
    if (!current_node) TT_ERROR("Compiler Error: Current node state is null");
    node_value_map[current_node] = val;
}

void CompilerVisitor::operator()(const InputOp& op) {
    auto func = llvm::dyn_cast<mlir::func::FuncOp>(module.getBody()->front());
    if (op.id >= func.getNumArguments()) {
        TT_ERROR("Compiler Error: InputOp ID exceeds generated function arguments.");
    }

    mlir::Value const arg = func.getArgument(op.id);
    set_mlir_value(InputOp{}, arg);
}

void CompilerVisitor::operator()(const ReluOp& op) {
    mlir::Location const loc = builder.getUnknownLoc(); // using unknown loc for now, its only debug information, change later perhaps
    // each op node carries with it a vector of ptrs to it's input nodes in the correct order (I sure hope so)
    auto input = get_mlir_value(current_node->inputs()[0]);
    auto type = input.getType().cast<mlir::RankedTensorType>();

    mlir::IntegerAttr const min_int = builder.getI64IntegerAttr(0);
    mlir::IntegerAttr const max_int = builder.getI64IntegerAttr(std::numeric_limits<int64_t>::max());
    mlir::FloatAttr const min_fp32 = builder.getF32FloatAttr(0.0f);
    mlir::FloatAttr const max_fp32 = builder.getF32FloatAttr(std::numeric_limits<float>::max());

    auto relu = builder.create<mlir::tosa::ClampOp>(loc, type, input, min_int, max_int, min_fp32, max_fp32);
    set_mlir_value(ReluOp{}, relu.getResult());
}

void CompilerVisitor::operator()(const AddOp& op) {
    mlir::Location const loc = builder.getUnknownLoc();

    auto lhs = get_mlir_value(current_node->inputs()[0]);
    auto rhs = get_mlir_value(current_node->inputs()[1]);
    auto type = lhs.getType();

    auto add = builder.create<mlir::tosa::AddOp>(loc, type, lhs, rhs);
    set_mlir_value(AddOp{}, add.getResult());
}

void CompilerVisitor::operator()(const BroadcastOp& op) {
    std::cout << "  [Compiler] BroadcastOp -> generating IR..." << std::endl;
}

void CompilerVisitor::operator()(const ReshapeOp& op) {
    mlir::Location const loc = builder.getUnknownLoc();

    auto input = get_mlir_value(current_node->inputs()[0]);

    std::vector<int64_t> new_shape_i64;
    const std::vector<int>& dims = op.target_shape.to_vec();
    for(int d : dims) new_shape_i64.push_back(d);

    auto new_shape_attr = builder.getDenseI64ArrayAttr(new_shape_i64);

    // Construct result type
    auto element_type = input.getType().cast<mlir::RankedTensorType>().getElementType();
    auto result_type = mlir::RankedTensorType::get(new_shape_i64, element_type);

    auto reshape = builder.create<mlir::tosa::ReshapeOp>(loc, result_type, input, new_shape_attr);
    set_mlir_value(ReshapeOp{}, reshape.getResult());
}

void CompilerVisitor::operator()(const MatMulOp& op) {

    mlir::Location const loc = builder.getUnknownLoc();

    auto lhs = get_mlir_value(current_node->inputs()[0]);
    auto rhs = get_mlir_value(current_node->inputs()[1]);

    // MatMul output shape can differ from input, so calculate the result type
    Shape result_shape = infer_shape(current_node);
    mlir::Type result_type = to_mlir_type(builder, result_shape, kF32);

    auto matmul = builder.create<mlir::tosa::MatMulOp>(loc, result_type, lhs, rhs);

    set_mlir_value(MatMulOp{}, matmul.getResult());
}

Tensor JITCompiler::compile(std::shared_ptr<OpNode> final_node) {
    std::cout << "--- JIT COMPILATION START ---\n";
    visited_nodes.clear();

    impl->module = mlir::ModuleOp::create(impl->builder ? impl->builder->getUnknownLoc() : mlir::UnknownLoc::get(&impl->context));
    impl->builder = std::make_unique<mlir::OpBuilder>(&impl->context);

    // Gather Inputs and Infer Output Shape
    std::vector<InputOp> const graph_inputs = get_sorted_graph_inputs(final_node);
    Shape const final_output_shape = infer_shape(final_node);

    std::vector<mlir::Type> arg_types;
    for (const auto& in_op : graph_inputs) {
        arg_types.push_back(to_mlir_type(*impl->builder, in_op.shape, in_op.dtype));
    }

    mlir::Type const result_type = to_mlir_type(*impl->builder, final_output_shape, kF32);

    const auto funcType = impl->builder->getFunctionType(arg_types, {result_type});
    auto function = mlir::func::FuncOp::create(impl->builder->getUnknownLoc(), "main_graph", funcType);
    function.addEntryBlock();
    impl->module.push_back(function);

    impl->builder->setInsertionPointToStart(&function.front());
    CompilerVisitor visitor(*impl->builder, impl->module, impl->context);

    if (final_node) {
        visit_recursive(final_node, visitor);
    }

    auto result_value = visitor.get_mlir_value(final_node);
    impl->builder->create<mlir::func::ReturnOp>(impl->builder->getUnknownLoc(), result_value);

    if (failed(mlir::verify(impl->module))) {
        impl->module.emitError("Module verification failed");
        TT_ERROR("MLIR Verification Failed");
    }

    std::cout << "--- GENERATED MLIR (TOSA) ---\n";
    impl->module.dump();
    std::cout << "-----------------------------\n";

    return full(0.0f, {1}, kCPU);
}

void JITCompiler::visit_recursive(const std::shared_ptr<OpNode>& node, CompilerVisitor& visitor) {
    auto key = reinterpret_cast<uintptr_t>(node.get());
    if (visited_nodes.contains(key)) return;

    for (const auto& input : node->inputs()) {
        if (input) visit_recursive(input, visitor);
    }

    visitor.set_current_node(node);
    visitor.dispatch(node->op());
    visited_nodes[key] = true;
}

} // namespace tinytensor::jit