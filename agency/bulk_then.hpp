/// \file
/// \brief Include this file to use bulk_then().
///

#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/control_structures/bulk_then_execution_policy.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/execution/execution_agent.hpp>
#include <agency/execution/executor/properties/detail/bulk_guarantee_depth.hpp>


namespace agency
{
namespace detail
{


template<bool enable, class ExecutionPolicy, class Function, class Future, class... Args>
struct enable_if_bulk_then_execution_policy_impl {};

template<class ExecutionPolicy, class Function, class Future, class... Args>
struct enable_if_bulk_then_execution_policy_impl<true, ExecutionPolicy, Function, Future, Args...>
{
  using type = bulk_then_execution_policy_result_t<ExecutionPolicy,Function,Future,Args...>;
};

template<class ExecutionPolicy, class Function, class Future, class... Args>
struct enable_if_bulk_then_execution_policy
  : enable_if_bulk_then_execution_policy_impl<
      is_bulk_then_possible_via_execution_policy<decay_t<ExecutionPolicy>,Function,Future,Args...>::value,
      decay_t<ExecutionPolicy>,
      Function,
      Future,
      Args...
    >
{};


} // end detail


/// \brief Creates a bulk continuation.
/// \ingroup control_structures
///
///
/// `bulk_then` is a control structure which asynchronously creates a group of function invocations with forward progress ordering as required by an execution policy.
/// These invocations are a *bulk continuation* to a predecessor bulk asynchronous invocation. The predecessor bulk asynchronous invocation is represented by a
/// future object, and the continuation will not execute until the predecessor's future object becomes ready.
/// The results of the continuation's invocations, if any, are collected into a container and returned as `bulk_then`'s asynchronous result.
/// A future object corresponding to the eventual availability of this container is returned as `bulk_then`'s result.
///
/// `bulk_then` asynchronously creates a group of function invocations of size `N`, and each invocation i in `[0,N)` has the following form:
///
///     result_i = f(agent_i, predecessor_arg, arg_i_1, arg_i_2, ..., arg_i_M)
///
/// `agent_i` is a reference to an **execution agent** which identifies the ith invocation within the group.
/// The parameter `predecessor_arg` is a reference to the value of the future object used as a parameter to `bulk_then`. If this future object has a `void` value, then this parameter is omitted.
/// The parameter `arg_i_j` depends on the `M` arguments `arg_j` passed to `bulk_invoke`:
///    * If `arg_j` is a **shared parameter**, then it is a reference to an object shared among all execution agents in `agent_i`'s group.
///    * Otherwise, `arg_i_j` is a copy of argument `arg_j`.
///
/// If the invocations of `f` do not return `void`, these results are collected and returned in a container `results`, whose type is implementation-defined.
/// If invocation i returns `result_i`, and this invocation's `agent_i` has index `idx_i`, then `results[idx_i]` yields `result_i`.
///
/// \param policy An execution policy describing the requirements of the execution agents created by this call to `bulk_then`.
/// \param f      A function defining the work to be performed by execution agents.
/// \param predecessor A future object representing the predecessor task. Its future value, if it has one, is passed to `f` as an argument when `f` is invoked.
///                    After `bulk_then` returns, `predecessor` is invalid if it is not a shared future.
/// \param args   Additional arguments to pass to `f` when it is invoked.
/// \return `void`, if `f` has no result; otherwise, a container of `f`'s results indexed by the execution agent which produced them.
///
/// \tparam ExecutionPolicy This type must fulfill the requirements of `ExecutionPolicy`.
/// \tparam Function `Function`'s first parameter type must be `ExecutionPolicy::execution_agent_type&`.
///         The types of its additional parameters must match `Args...`.
/// \tparam Future This type must fulfill the requirements of `Future`. If the value type of this `Future` is not `void`, this type
///         must match the type of the second parameter of `Function`.
/// \tparam Args Each type in `Args...` must match the type of the corresponding parameter of `Function`.
///
/// The following example demonstrates how to use `bulk_then` to sequence a continuation after a predecessor task:
///
/// \include hello_then.cpp
///
/// Messages from agents in the predecessor task are guaranteed to be output before messages from the continuation:
///
/// ~~~~
/// $ clang -std=c++11 -I. -lstdc++ -pthread examples/hello_then.cpp -o hello_then
/// $ ./hello_then
/// Starting predecessor and continuation tasks asynchronously...
/// Sleeping before waiting on the continuation...
/// Hello, world from agent 0 in the predecessor task
/// Hello, world from agent 1 in the predecessor task
/// Hello, world from agent 2 in the predecessor task
/// Hello, world from agent 3 in the predecessor task
/// Hello, world from agent 4 in the predecessor task
/// Hello, world from agent 0 in the continuation
/// Hello, world from agent 1 in the continuation
/// Hello, world from agent 2 in the continuation
/// Hello, world from agent 3 in the continuation
/// Hello, world from agent 4 in the continuation
/// Woke up, waiting for the continuation to complete...
/// OK
/// ~~~~
///
/// \see bulk_invoke
/// \see bulk_async
template<class ExecutionPolicy, class Function, class Future, class... Args>
__AGENCY_ANNOTATION
#ifndef DOXYGEN_SHOULD_SKIP_THIS
typename detail::enable_if_bulk_then_execution_policy<
  ExecutionPolicy, Function, Future, Args...
>::type
#else
see_below
#endif
  bulk_then(ExecutionPolicy&& policy, Function f, Future& predecessor, Args&&... args)
{
  using agent_traits = execution_agent_traits<typename std::decay<ExecutionPolicy>::type::execution_agent_type>;
  const size_t num_shared_params_for_agent = detail::bulk_guarantee_depth<typename agent_traits::execution_requirement>::value;

  return detail::bulk_then_execution_policy(
    detail::index_sequence_for<Args...>(),
    detail::make_index_sequence<num_shared_params_for_agent>(),
    policy,
    f,
    predecessor,
    std::forward<Args>(args)...
  );
}


} // end agency

