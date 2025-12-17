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

struct ReshapeOp {
    Shape target_shape;
};

// The Variant
using OpType = std::variant<
    InputOp,
    ReluOp,
    AddOp,
    BroadcastOp,
    ReshapeOp
>;

} // namespace tinytensor::jit