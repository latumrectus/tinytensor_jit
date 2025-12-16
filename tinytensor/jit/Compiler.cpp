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

JITCompiler::JITCompiler() : impl_(std::make_unique<Impl>()) {}
JITCompiler::~JITCompiler() = default;
JITCompiler::JITCompiler(JITCompiler&&) noexcept = default;
JITCompiler& JITCompiler::operator=(JITCompiler&&) noexcept = default;
CompilerVisitor::CompilerVisitor(mlir::OpBuilder& b, mlir::ModuleOp& m, mlir::MLIRContext& c)
    : builder_(b), module_(m), context_(c) {}

mlir::Value CompilerVisitor::get_mlir_value(const std::shared_ptr<OpNode>& node) const {
    if (!node_value_map.contains(node)) {
        TT_ERROR("Compiler Error: Node dependency not found in SSA map.");
    }
    return node_value_map.at(node);
}

void CompilerVisitor::set_mlir_value(const OpType&, mlir::Value val) {
    if (!current_node_) TT_ERROR("Compiler Error: Current node state is null");
    node_value_map[current_node_] = val;
}

void CompilerVisitor::operator()(const InputOp& op) {
    auto func = llvm::dyn_cast<mlir::func::FuncOp>(module_.getBody()->front());
    if (op.id >= func.getNumArguments()) {
        TT_ERROR("Compiler Error: InputOp ID exceeds generated function arguments.");
    }

    mlir::Value const arg = func.getArgument(op.id);
    set_mlir_value(InputOp{}, arg);
}

void CompilerVisitor::operator()(const ReluOp& op) {
    std::cout << "  [Compiler] ReluOp -> generating IR..." << std::endl;
}

void CompilerVisitor::operator()(const AddOp& op) {
    std::cout << "  [Compiler] AddOp -> generating IR..." << std::endl;
}

void CompilerVisitor::operator()(const BroadcastOp& op) {
    std::cout << "  [Compiler] BroadcastOp -> generating IR..." << std::endl;
}

void CompilerVisitor::operator()(const ReshapeOp& op) {
    std::cout << "  [Compiler] ReshapeOp -> generating IR..." << std::endl;
}

Tensor JITCompiler::compile(std::shared_ptr<OpNode> final_node) {
    std::cout << "--- JIT COMPILATION START (Gazprea Structure) ---" << std::endl;
    visited_nodes_.clear();

    // Initialize MLIR State
    // future...
    // mlir::OpBuilder builder(&impl_->context);
    // mlir::ModuleOp module = mlir::ModuleOp::create(...)

    // We pass nullptr because haven't linked MLIR yet,
    // but the visitor structure is ready to receive them.
    CompilerVisitor visitor(nullptr, nullptr, nullptr);

    // run Traversal
    if (final_node) {
        visit_recursive(final_node, visitor);
    }

    std::cout << "--- JIT COMPILATION DONE ---" << std::endl;

    return full(0.0f, {1}, kCPU);
}

void JITCompiler::visit_recursive(const std::shared_ptr<OpNode>& node, CompilerVisitor& visitor) {
    auto key = reinterpret_cast<uintptr_t>(node.get());
    if (visited_nodes_.count(key)) return;

    // Post-order traversal
    for (const auto& input : node->inputs()) {
        if (input) visit_recursive(input, visitor);
    }

    // visit current
    visitor.dispatch(node->op());
    visited_nodes_[key] = true;
}

} // namespace tinytensor::jit