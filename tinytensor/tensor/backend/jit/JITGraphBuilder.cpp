#include "JITGraphBuilder.h"

#include <tt/jit/Graph.h>
#include <tt/jit/Ops.h>
#include <tt/shape.h>

#include "storage_jit.h"

namespace tinytensor {

// manually pass Shape and Dtype because StorageBase doesn't hold metadata.
Tensor make_jit_tensor(std::shared_ptr<jit::OpNode> node, Shape shape, ScalarType dtype) {
    // create the JIT storage holding the node
    auto storage = std::make_shared<StorageJIT>(std::move(node));

    // use the Tensor constructor designed for backends:
    // Tensor(std::shared_ptr<StorageBase> storage, ScalarType dtype, Shape shape, Device device);
    return Tensor(std::move(storage), dtype, shape, kJIT);
}

Tensor JITGraphBuilder::relu(const Tensor &tensor) const {
    auto& in_jit = tensor.get_storage<StorageJIT>();

    auto node = jit::GetGlobalGraph().create_node(jit::ReluOp{}, {in_jit.get_node()});

    return make_jit_tensor(node, tensor.shape(), tensor.dtype());
}

Tensor JITGraphBuilder::add(const Tensor &lhs, const Tensor &rhs) const {
    auto& lhs_jit = lhs.get_storage<StorageJIT>();
    auto& rhs_jit = rhs.get_storage<StorageJIT>();

    // assume shapes match or that the user has handled broadcasting explicitly, as per readme.md
    // (Future TODO: Handle explicit views where Tensor shape != OpNode shape)

    auto node = jit::GetGlobalGraph().create_node(
        jit::AddOp{},
        {lhs_jit.get_node(), rhs_jit.get_node()}
    );

    // take the LHS shape/dtype as the truth
    return make_jit_tensor(node, lhs.shape(), lhs.dtype());
}
auto JITGraphBuilder::full(const Scalar &value, const Shape &shape, int device_id) const -> StoragePtr {
    static int global_input_id = 0;
    unsigned int const id = global_input_id++;

    auto node = jit::GetGlobalGraph().create_node(jit::InputOp{.id=id, .shape=shape, .dtype=kF32}, {});
    return std::make_unique<StorageJIT>(std::move(node));
}
auto JITGraphBuilder::full(const Scalar &value, std::size_t N, int device_id) const -> StoragePtr {
    static int global_input_id = 0;
    unsigned int const id = global_input_id++;

    // Note: BackendBase::full only gives N, not the Shape
    // create a 1D shape for the node, frontend Tensor will handle the view
    Shape const shape = {(int)N};

    constexpr ScalarType dtype = kF32;

    auto node = jit::GetGlobalGraph().create_node(
        jit::InputOp{.id=id, .shape=shape, .dtype=dtype},
        {}
    );

    return std::make_unique<StorageJIT>(std::move(node));
}

auto JITGraphBuilder::batched_matmul(const Tensor &lhs, const Tensor &rhs) const -> Tensor {
    // TODO: Handle compatibility/legality checks for MatMul.
    auto& lhs_jit = lhs.get_storage<StorageJIT>();
    auto rhs_jit = rhs.get_storage<StorageJIT>();

    std::shared_ptr<jit::OpNode> lhs_node = lhs_jit.get_node();
    std::shared_ptr<jit::OpNode> rhs_node = rhs_jit.get_node();

    Shape lhs_shape = lhs.shape();
    Shape rhs_shape = rhs.shape();

    bool promoted_lhs = false;
    bool promoted_rhs = false;

    // Promote LHS or RHS to 3D tensor[Rows, Cols] -> [1, Rows, Cols]

    if (lhs_shape.ndim() == 2) {
        Shape new_shape = {1, lhs_shape[0], lhs_shape[1]};
        lhs_node = jit::GetGlobalGraph().create_node(
            jit::ReshapeOp{new_shape},
            {lhs_node}
        );
        lhs_shape = new_shape;
        promoted_lhs = true;
    }

    if (rhs_shape.ndim() == 2) {
        Shape new_shape = {1, rhs_shape[0], rhs_shape[1]};
        rhs_node = jit::GetGlobalGraph().create_node(
            jit::ReshapeOp{new_shape},
            {rhs_node}
        );
        rhs_shape = new_shape;
        promoted_rhs = true;
    }

    // TODO: Handle Broadcasting (or not if I stick to tuero's philosophy) if batch dimensions differ ([1, R, C] vs [N, R, C])
    // TOSA matmul supports broadcasting on batch dim implicitly if shapes match rules (might wanna throw an error instead we shall see)

    // now we have MatMulOp in the form of [Batch, M, K] x [Batch, K, N] (where batch may be 1)
    auto matmul_node = jit::GetGlobalGraph().create_node(
        jit::MatMulOp{},
        {lhs_node, rhs_node}
    );

    Shape out_shape_3d = {lhs_shape[0], lhs_shape[1], rhs_shape[2]};

    // result back to 2D if both inputs were originally 2D
    if (promoted_lhs && promoted_rhs) {
        Shape out_shape_2d = {out_shape_3d[1], out_shape_3d[2]};
        auto reshape_back_node = jit::GetGlobalGraph().create_node(
            jit::ReshapeOp{out_shape_2d},
            {matmul_node}
        );
        return make_jit_tensor(reshape_back_node, out_shape_2d, lhs.dtype());
    }

    return make_jit_tensor(matmul_node, out_shape_3d, lhs.dtype());
}

auto JITGraphBuilder::Broadcast(const Tensor &tensor, const Shape &targetShape) const{
    auto& input_tensor = tensor.get_storage<StorageJIT>();
    auto node = jit::GetGlobalGraph().create_node(
        jit::BroadcastOp{.target_shape = targetShape},
        {input_tensor.get_node()}
    );

    return make_jit_tensor(node, targetShape, tensor.dtype());
}


} // namespace tinytensor