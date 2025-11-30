#include <tt/jit/Compiler.h>
#include <tt/jit/OpNode.h>
#include <tt/jit/Ops.h>
#include <tt/tensor.h>
#include <iostream>

// add actual mlir stuff here

namespace tinytensor::jit {

// This hides the actual MLIR objects from the header
struct JITCompiler::Impl {
    // In the future:
    // mlir::MLIRContext context;
    // mlir::ModuleOp module;
    // std::unique_ptr<mlir::OpBuilder> builder;

    // For now, placeholders:
    int mock_context = 0;
};


CompilerVisitor::CompilerVisitor(mlir::OpBuilder* builder,
                                 mlir::ModuleOp* module,
                                 mlir::MLIRContext* context)
    : builder_(builder), module_(module), context_(context) {
    // In the future:
    // builder_->setInsertionPointToStart(module_->getBody());
}

void CompilerVisitor::operator()(const InputOp& op) {

    std::cout << "  [Compiler] InputOp (ID: " << op.id << ") -> generating IR..." << std::endl;
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



JITCompiler::JITCompiler() : impl_(std::make_unique<Impl>()) {}
JITCompiler::~JITCompiler() = default;

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