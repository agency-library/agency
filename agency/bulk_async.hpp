/// \file
/// \brief Include this file to use bulk_async().
///

#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/control_structures/bulk_async_execution_policy.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/control_structures/is_bulk_call_possible_via_execution_policy.hpp>
#include <agency/execution/execution_agent.hpp>
#include <agency/execution/executor/properties/detail/bulk_guarantee_depth.hpp>


namespace agency
{
namespace detail
{


template<bool enable, class ExecutionPolicy, class Function, class... Args>
struct enable_if_bulk_async_execution_policy_impl {};

template<class ExecutionPolicy, class Function, class... Args>
struct enable_if_bulk_async_execution_policy_impl<true, ExecutionPolicy, Function, Args...>
{
  using type = bulk_async_execution_policy_result_t<ExecutionPolicy,Function,Args...>;
};

template<class ExecutionPolicy, class Function, class... Args>
struct enable_if_bulk_async_execution_policy
  : enable_if_bulk_async_execution_policy_impl<
      is_bulk_call_possible_via_execution_policy<decay_t<ExecutionPolicy>,Function,Args...>::value,
      decay_t<ExecutionPolicy>,
      Function,
      Args...
    >
{};


} // end detail


/// \brief Creates a bulk asynchronous invocation.
/// \ingroup control_structures
///
///
/// `bulk_async` is a control structure which asynchronously creates a group of function invocations with forward progress ordering as required by an execution policy.
/// The results of these invocations, if any, are collected into a container and returned as `bulk_async`'s asynchronous result.
/// A future object corresponding to the eventual availability of this container is returned as `bulk_async`'s result.
///
/// `bulk_async` asynchronously creates a group of function invocations of size `N`, and each invocation i in `[0,N)` has the following form:
///
///     result_i = f(agent_i, arg_i_1, arg_i_2, ..., arg_i_M)
///
/// `agent_i` is a reference to an **execution agent** which identifies the ith invocation within the group.
/// The parameter `arg_i_j` depends on the `M` arguments `arg_j` passed to `bulk_async`:
///    * If `arg_j` is a **shared parameter**, then it is a reference to an object shared among all execution agents in `agent_i`'s group.
///    * Otherwise, `arg_i_j` is a copy of argument `arg_j`.
///
/// If the invocations of `f` do not return `void`, these results are collected and returned in a container `results`, whose type is implementation-defined.
/// If invocation i returns `result_i`, and this invocation's `agent_i` has index `idx_i`, then `results[idx_i]` yields `result_i`.
///
/// \param policy An execution policy describing the requirements of the execution agents created by this call to `bulk_async`.
/// \param f      A function defining the work to be performed by execution agents.
/// \param args   Additional arguments to pass to `f` when it is invoked.
/// \return A `void` future object, if `f` has no result; otherwise, a future object corresponding to the eventually available container of `f`'s results indexed by the execution agent which produced them.
/// \note The type of future object returned by `bulk_async` is a property of the type of `ExecutionPolicy` used as a parameter.
///
/// \tparam ExecutionPolicy This type must fulfill the requirements of `ExecutionPolicy`.
/// \tparam Function `Function`'s first parameter type must be `ExecutionPolicy::execution_agent_type&`.
///         The types of its additional parameters must match `Args...`.
/// \tparam Args Each type in `Args...` must match the type of the corresponding parameter of `Function`.
///
/// The following example demonstrates how to use `bulk_async` to create tasks which execute asynchronously with the caller.
///
/// \include hello_async.cpp
///
/// Messages from the agents in the two asynchronous tasks are printed while the main thread sleeps:
///
/// ~~~~
/// $ clang -std=c++11 -I. -lstdc++ -pthread examples/hello_async.cpp -o hello_async
/// $ ./hello_async
/// Starting two tasks asynchronously...
/// Sleeping before waiting on the tasks...
/// Hello, world from agent 0 in task 1
/// Hello, world from agent 1 in task 1
/// Hello, world from agent 2 in task 1
/// Hello, world from agent 3 in task 1
/// Hello, world from agent 4 in task 1
/// Hello, world from agent 0 in task 2
/// Hello, world from agent 1 in task 2
/// Hello, world from agent 2 in task 2
/// Hello, world from agent 3 in task 2
/// Hello, world from agent 4 in task 2
/// Woke up, waiting for the tasks to complete...
/// OK
/// ~~~~
///
/// \see bulk_invoke
/// \see bulk_then
template<class ExecutionPolicy, class Function, class... Args>
__AGENCY_ANNOTATION
#ifndef DOXYGEN_SHOULD_SKIP_THIS
typename detail::enable_if_bulk_async_execution_policy<
  ExecutionPolicy, Function, Args...
>::type
#else
see_below
#endif
  bulk_async(ExecutionPolicy&& policy, Function f, Args&&... args)
{
  using agent_traits = execution_agent_traits<typename std::decay<ExecutionPolicy>::type::execution_agent_type>;
  const size_t num_shared_params = detail::bulk_guarantee_depth<typename agent_traits::execution_requirement>::value;

  return detail::bulk_async_execution_policy(detail::index_sequence_for<Args...>(), detail::make_index_sequence<num_shared_params>(), policy, f, std::forward<Args>(args)...);
}


} // end agency

