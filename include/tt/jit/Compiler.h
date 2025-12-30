#pragma once
#include <memory>
#include <unordered_map>
#include <iostream>
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

class OpNode;

class CompilerVisitor : public BaseVisitor<CompilerVisitor> {
public:
    using BaseVisitor::operator();

    CompilerVisitor(mlir::OpBuilder& builder,
                    mlir::ModuleOp& module,
                    mlir::MLIRContext& context);

    // Visitor Methods
    void operator()(const InputOp& op);
    void operator()(const ReluOp& op);
    void operator()(const AddOp& op);
    void operator()(const BroadcastOp& op);
    void operator()(const ReshapeOp& op);
    void operator()(const MatMulOp &op);

    mlir::Value get_mlir_value(const std::shared_ptr<OpNode>& node) const;
    void set_mlir_value(const OpType& op_variant, mlir::Value val);
    void set_current_node(const std::shared_ptr<OpNode>& node) { current_node = node; }

private:
    mlir::OpBuilder& builder;
    mlir::ModuleOp& module;
    mlir::MLIRContext& context;

    std::unordered_map<std::shared_ptr<OpNode>, mlir::Value> node_value_map;

    std::shared_ptr<OpNode> current_node;

    friend class JITCompiler;
};

class JITCompiler {
public:
    JITCompiler();
    ~JITCompiler();
    JITCompiler(JITCompiler&&) noexcept;
    JITCompiler& operator=(JITCompiler&&) noexcept;

    JITCompiler(const JITCompiler&) = delete;
    JITCompiler& operator=(const JITCompiler&) = delete;

    Tensor compile(std::shared_ptr<OpNode> final_node);

    int lowerDialects();

    void dumpLLVM(std::ostream &os);

private:
    void visit_recursive(const std::shared_ptr<OpNode>& node, CompilerVisitor& visitor);

    std::unordered_map<uintptr_t, bool> visited_nodes;

    // We hold these as opaque pointers or unique_ptrs to implementation classes
    // to keep dependencies out of the header until Step 5.
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace tinytensor::jit