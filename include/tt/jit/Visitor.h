//
// Created by ishaan on 11/29/25.
//

#pragma once

#include <tt/jit/Ops.h>
#include <tt/exception.h>
#include <variant>
#include <iostream>

//Helper for CRTP classes

/* BaseVisitor for ast with default traversals
 * // std::visit does compile-time dispatch through some template magic
 * Usage: std::visit(BaseVisitor, node);
 *
 * Overriding:
 *
 * struct DerivedVisitor : public BaseVisitor<DerivedVisitor> {
 *     // IMPORTANT: Bring default traversals into scope, or else you'll have to override all of them
 *     using BaseVisitor::operator()
 *
 *     // Override required operators.
 *     std::any operator()(std::shared_ptr<BlockStat> block).
 *     ...
 * }
 */
namespace tinytensor::jit {

//CRTP Helper 
template <typename Derived>
struct CRTP {
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }
};

// Base Visitor
template <typename Derived>
struct BaseVisitor : CRTP<Derived> {
    
   
    void dispatch(const OpType& op) {
        std::visit(this->derived(), op);
    }

    // default impls
    // The Derived class will "using BaseVisitor::operator()" to inherit these.
    // This forces the Derived class to strictly implement only what it needs,
    // but crashes if it misses something it SHOULD handle.

    void operator()(const InputOp& op) {
        TT_ERROR("JIT Visitor: Unhandled InputOp");
    }

    void operator()(const ReluOp& op) {
        TT_ERROR("JIT Visitor: Unhandled ReluOp");
    }

    void operator()(const AddOp& op) {
        TT_ERROR("JIT Visitor: Unhandled AddOp");
    }

    void operator()(const BroadcastOp& op) {
        TT_ERROR("JIT Visitor: Unhandled BroadcastOp");
    }
    
    void operator()(const ReshapeOp& op) {
        TT_ERROR("JIT Visitor: Unhandled ReshapeOp");
    }

    template <typename T>
    void operator()(const T& op) {
        TT_ERROR("JIT Visitor: Unknown Operation Type");
    }
};

} // namespace tinytensor::jit
