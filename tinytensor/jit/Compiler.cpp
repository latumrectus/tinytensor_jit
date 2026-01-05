#include <tt/jit/Compiler.h>
#include <tt/jit/OpNode.h>
#include <tt/jit/Ops.h>
#include <tt/tensor.h>
#include <algorithm>
#include <iostream>

// MLIR Core
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

// MLIR Dialects
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

// MLIR Passes & Conversion
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToSCF/TosaToSCF.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

// LLVM Translation
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_os_ostream.h"

namespace tinytensor::jit {

// This hides the actual MLIR objects from the header
struct JITCompiler::Impl {
    mlir::MLIRContext context;
    mlir::ModuleOp module;
    std::unique_ptr<mlir::OpBuilder> builder;

    // LLVM Context
    llvm::LLVMContext llvm_context;
    std::unique_ptr<llvm::Module> llvm_module;

    Impl() {
        // Load necessary dialects
        context.loadDialect<mlir::func::FuncDialect>();
        context.loadDialect<mlir::tosa::TosaDialect>();
        context.loadDialect<mlir::arith::ArithDialect>();
        context.loadDialect<mlir::scf::SCFDialect>();
        context.loadDialect<mlir::cf::ControlFlowDialect>();
        context.loadDialect<mlir::memref::MemRefDialect>();
        context.loadDialect<mlir::LLVM::LLVMDialect>();
        context.loadDialect<mlir::bufferization::BufferizationDialect>();
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

    // // DEBUG PRINT
    // std::cout << "Debug to_mlir_type: ";
    // for(auto d : dims) std::cout << d << " ";
    // std::cout << std::endl;

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
    mlir::Location loc = builder.getUnknownLoc();

    auto input = get_mlir_value(current_node->inputs()[0]); // get the input tensor
    auto in_type = input.getType().cast<mlir::RankedTensorType>(); // convert to ranked tensor type

    llvm::ArrayRef<int64_t> in_shape = in_type.getShape(); // convert shape to long array as MLIR prefers it

    const std::vector<int>& target_dims_int = op.target_shape.to_vec();
    std::vector<int64_t> target_shape(target_dims_int.begin(), target_dims_int.end());

    if (in_shape.size() < target_shape.size()) {
        std::vector<int64_t> expanded_shape;

        // Right-align the dimensions
        size_t offset = target_shape.size() - in_shape.size();
        for (size_t i = 0; i < offset; ++i) expanded_shape.push_back(1);
        for (int64_t dim : in_shape) expanded_shape.push_back(dim);

        // Create Reshape Op
        auto new_type = mlir::RankedTensorType::get(expanded_shape, in_type.getElementType());
        auto new_shape_attr = builder.getDenseI64ArrayAttr(expanded_shape);
        auto reshape = builder.create<mlir::tosa::ReshapeOp>(loc, new_type, input, new_shape_attr);

        // Update input to point to the reshaped result
        input = reshape.getResult();
        in_shape = input.getType().cast<mlir::RankedTensorType>().getShape();
    }

    // Compute Tiling Multiples
    std::vector<int64_t> multiples;
    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (in_shape[i] == target_shape[i]) {
            multiples.push_back(1);
        } else if (in_shape[i] == 1) {
            multiples.push_back(target_shape[i]);
        } else {
            TT_ERROR("Broadcast compatibility error: Cannot broadcast dimension " +
                     std::to_string(in_shape[i]) + " to " + std::to_string(target_shape[i]));
        }
    }

    // Create Tile Op
    auto multiples_attr = builder.getDenseI64ArrayAttr(multiples);
    auto result_type = mlir::RankedTensorType::get(target_shape, in_type.getElementType());

    auto tile = builder.create<mlir::tosa::TileOp>(loc, result_type, input, multiples_attr);

    set_mlir_value(BroadcastOp{}, tile.getResult());

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

    return full(0.0f, {1}, kCPU);
}

int JITCompiler::lowerDialects() {
    mlir::PassManager pm(&impl->context);

    // TOSA -> Linalg/SCF
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalgNamed());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToArith());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToSCF());

    // Bufferization (Tensor -> MemRef)
    pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
    pm.addPass(mlir::bufferization::createOneShotBufferizePass());

    // Backend Lowering
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    if (mlir::failed(pm.run(impl->module))) {
        llvm::errs() << "MLIR Pass pipeline failed\n";
        impl->module.dump();
        return 1;
    }
    return 0;
}

void JITCompiler::dumpLLVM(std::ostream &os) {

    // Register translations
    mlir::registerBuiltinDialectTranslation(impl->context);
    mlir::registerLLVMDialectTranslation(impl->context);

    // Translate to LLVM IR
    impl->llvm_module = mlir::translateModuleToLLVMIR(impl->module, impl->llvm_context);

    if (!impl->llvm_module) {
        llvm::errs() << "Failed to translate module to LLVM IR\n";
        return;
    }

    llvm::raw_os_ostream output(os);
    output << *impl->llvm_module;
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