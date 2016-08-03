#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/control_structures/bulk_then_execution_policy.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/execution/execution_agent.hpp>

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


template<class ExecutionPolicy, class Function, class Future, class... Args>
typename detail::enable_if_bulk_then_execution_policy<
  ExecutionPolicy, Function, Future, Args...
>::type
  bulk_then(ExecutionPolicy&& policy, Function f, Future& fut, Args&&... args)
{
  using agent_traits = execution_agent_traits<typename std::decay<ExecutionPolicy>::type::execution_agent_type>;
  const size_t num_shared_params_for_agent = detail::execution_depth<typename agent_traits::execution_category>::value;

  return detail::bulk_then_execution_policy(
    detail::index_sequence_for<Args...>(),
    detail::make_index_sequence<num_shared_params_for_agent>(),
    policy,
    f,
    fut,
    std::forward<Args>(args)...
  );
}


} // end agency

