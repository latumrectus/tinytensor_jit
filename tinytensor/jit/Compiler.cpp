#include <tt/jit/Compiler.h>
#include <tt/jit/OpNode.h>
#include <tt/jit/Ops.h>
#include <tt/tensor.h>
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
    std::cout << "  [Compiler] ReshapeOp -> generating IR..." << std::endl;
}

Tensor JITCompiler::compile(std::shared_ptr<OpNode> final_node) {
    std::cout << "--- JIT COMPILATION START ---\n";
    visited_nodes.clear();

    impl->module = mlir::ModuleOp::create(impl->builder ? impl->builder->getUnknownLoc() : mlir::UnknownLoc::get(&impl->context));
    impl->builder = std::make_unique<mlir::OpBuilder>(&impl->context);

    std::vector<int64_t> shape = {2, 2};
    auto tensorType = mlir::RankedTensorType::get(shape, impl->builder->getF32Type());

    auto funcType = impl->builder->getFunctionType({tensorType, tensorType}, {tensorType});
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