#pragma once
#include <memory>
#include <unordered_map>
#include <string>
#include <tt/tensor.h>
#include <tt/jit/OpNode.h>
#include <tt/jit/Visitor.h>

namespace mlir {
    class MLIRContext;
    class OpBuilder;
    class ModuleOp;
    class Value;
    class Location;
}

namespace tinytensor::jit {

/**
 * @class CompilerVisitor
 * @brief The heavy lifter. Translates OpNodes into MLIR/LLVM IR.
 * Structure mimics Gazprea's BackendVisitor.
 */
class CompilerVisitor : public BaseVisitor<CompilerVisitor> {
public:
    using BaseVisitor::operator();

    // Constructor accepts the Compiler State
    // (We use raw pointers for now to avoid unique_ptr issues with incomplete types)
    CompilerVisitor(mlir::OpBuilder* builder,
                   mlir::ModuleOp* module,
                   mlir::MLIRContext* context);

    // Visitor Methods
    void operator()(const InputOp& op);
    void operator()(const ReluOp& op);
    void operator()(const AddOp& op);
    void operator()(const BroadcastOp& op);
    void operator()(const ReshapeOp& op);

    // mlir::Value resolveValue(OpNode* node);

private:
    mlir::OpBuilder* builder_;
    mlir::ModuleOp* module_;
    mlir::MLIRContext* context_;

    // Tracks the current compilation node to map OpNode -> mlir::Value
    // std::unordered_map<OpNode*, mlir::Value> value_map_;
};


/**
 * @class JITCompiler
 * @brief The Driver. Manages the lifetime of MLIR Contexts and triggers the Visitor.
 */
class JITCompiler {
public:
    JITCompiler();
    ~JITCompiler();

    Tensor compile(std::shared_ptr<OpNode> final_node);

private:
    void visit_recursive(const std::shared_ptr<OpNode>& node, CompilerVisitor& visitor);

    std::unordered_map<uintptr_t, bool> visited_nodes_;

    // We hold these as opaque pointers or unique_ptrs to implementation classes
    // to keep dependencies out of the header until Step 5.
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tinytensor::jit