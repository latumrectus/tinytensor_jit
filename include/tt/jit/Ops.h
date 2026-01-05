//
// Created by ishaan on 11/27/25.
//

#pragma once
#include <variant>
#include <vector>
#include <cstdint>
#include <tt/shape.h>

namespace tinytensor::jit {

// Leaf node
struct InputOp {
    unsigned int id;
    Shape shape;
    ScalarType dtype;
};

// Operations
struct ReluOp {};
struct AddOp {};

struct BroadcastOp {
    Shape target_shape;
};

struct Conv2dOp {
    std::vector<int> stride;
    std::vector<int> padding;
    std::vector<int> dilation;
};

struct ReshapeOp {
    Shape target_shape;
};
struct MatMulOp {};

// The Variant
using OpType = std::variant<
    InputOp,
    ReluOp,
    AddOp,
    BroadcastOp,
    ReshapeOp,
    MatMulOp,
    Conv2dOp
>;

} // namespace tinytensor::jit