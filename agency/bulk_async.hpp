#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/control_structures/bulk_async_execution_policy.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/control_structures/is_bulk_call_possible_via_execution_policy.hpp>
#include <agency/execution/execution_agent.hpp>

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


template<class ExecutionPolicy, class Function, class... Args>
typename detail::enable_if_bulk_async_execution_policy<
  ExecutionPolicy, Function, Args...
>::type
  bulk_async(ExecutionPolicy&& policy, Function f, Args&&... args)
{
  using agent_traits = execution_agent_traits<typename std::decay<ExecutionPolicy>::type::execution_agent_type>;
  const size_t num_shared_params = detail::execution_depth<typename agent_traits::execution_category>::value;

  return detail::bulk_async_execution_policy(detail::index_sequence_for<Args...>(), detail::make_index_sequence<num_shared_params>(), policy, f, std::forward<Args>(args)...);
}


} // end agency

