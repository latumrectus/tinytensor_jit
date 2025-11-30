//
// Created by ishaan on 11/29/25.
//

#pragma once
#include <memory>
#include <unordered_map>
#include <tt/tensor.h>

namespace tinytensor::jit {

class OpNode;

class JITCompiler {
public:
    // Returns tinytensor::Tensor explicitly
    Tensor compile(std::shared_ptr<OpNode> final_node);

private:
    void visit(const std::shared_ptr<OpNode>& node);

    std::unordered_map<uintptr_t, bool> visited_nodes_;
};

} // namespace tinytensor::jit