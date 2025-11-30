//
// Created by ishaan on 11/29/25.
//
#include <tt/jit/Compiler.h>
#include <tt/jit/OpNode.h>
#include <tt/jit/Ops.h>
#include <tt/jit/Visitor.h>
#include <tt/tensor.h>
#include <iostream>
#include <variant>

namespace tinytensor::jit {

// Inherits from BaseVisitor
struct CompilerVisitor : BaseVisitor<CompilerVisitor> {

    using BaseVisitor::operator();

    void operator()(const InputOp& op) {
        std::cout << "  [Compiler] Found Input (ID: " << op.id << ")" << std::endl;
    }

    void operator()(const ReluOp& op) {
        std::cout << "  [Compiler] Compiling ReLU" << std::endl;
    }

    void operator()(const AddOp& op) {
        std::cout << "  [Compiler] Compiling Add" << std::endl;
    }

    void operator()(const BroadcastOp& op) {
        std::cout << "  [Compiler] Compiling Broadcast -> " << op.target_shape << std::endl;
    }

    void operator()(const ReshapeOp& op) {
        std::cout << "  [Compiler] Compiling Reshape -> " << op.target_shape << std::endl;
    }
};


Tensor JITCompiler::compile(std::shared_ptr<OpNode> final_node) {
    std::cout << "--- JIT COMPILATION START ---" << std::endl;

    visited_nodes_.clear();

    if (final_node) {
        visit(final_node);
    }

    std::cout << "--- JIT COMPILATION DONE ---" << std::endl;

    return full(0.0f, {1}, kCPU);
}

void JITCompiler::visit(const std::shared_ptr<OpNode>& node) {
    auto key = reinterpret_cast<uintptr_t>(node.get());
    if (visited_nodes_.count(key)) {
        return;
    }

    // Post-order traversal, visit inputs first
    for (const auto& input : node->inputs()) {
        if (input) {
            visit(input);
        }
    }

    // Process current node
    CompilerVisitor visitor;
    visitor.dispatch(node->op());

    visited_nodes_[key] = true;
}

} // namespace tinytensor::jit