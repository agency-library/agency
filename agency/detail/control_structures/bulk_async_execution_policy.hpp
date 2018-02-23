#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/control_structures/executor_functions/bulk_async_with_executor.hpp>
#include <agency/detail/control_structures/execute_agent_functor.hpp>
#include <agency/detail/control_structures/single_result.hpp>
#include <agency/detail/control_structures/bulk_invoke_execution_policy.hpp>
#include <agency/detail/control_structures/shared_parameter.hpp>
#include <agency/detail/control_structures/tuple_of_agent_shared_parameter_factories.hpp>
#include <agency/execution/execution_agent.hpp>
#include <agency/execution/executor/executor_traits/executor_shape.hpp>
#include <agency/execution/executor/executor_traits/detail/executor_barrier_types_as_scoped_in_place_type.hpp>
#include <agency/execution/execution_policy.hpp>
#include <agency/tuple.hpp>
#include <utility>

namespace agency
{
namespace detail
{


template<class ExecutionPolicy, class Function, class... Args>
struct bulk_async_execution_policy_result
{
  using type = execution_policy_future_t<
    ExecutionPolicy,
    bulk_invoke_execution_policy_result_t<ExecutionPolicy,Function,Args...>
  >;
};

template<class ExecutionPolicy, class Function, class... Args>
using bulk_async_execution_policy_result_t = typename bulk_async_execution_policy_result<ExecutionPolicy,Function,Args...>::type;


template<size_t... UserArgIndices, size_t... SharedArgIndices, class ExecutionPolicy, class Function, class... Args>
__AGENCY_ANNOTATION
bulk_async_execution_policy_result_t<
  ExecutionPolicy, Function, Args...
>
  bulk_async_execution_policy(index_sequence<UserArgIndices...>,
                              index_sequence<SharedArgIndices...>,
                              ExecutionPolicy& policy, Function f, Args&&... args)
{
  using agent_type = typename ExecutionPolicy::execution_agent_type;
  using agent_traits = execution_agent_traits<agent_type>;

  // get the parameters of the agent
  auto param = policy.param();
  auto agent_shape = agent_traits::domain(param).shape();

  using executor_type = typename ExecutionPolicy::executor_type;

  // get a list of barrier types to create as a scoped_in_place_type_t
  executor_barrier_types_as_scoped_in_place_type_t<executor_type> barriers;

  // this is a tuple of factories
  // each factory in the tuple creates the execution agent's shared parameter at the corresponding hierarchy level
  auto tuple_of_factories = detail::make_tuple_of_agent_shared_parameter_factories<agent_type>(param, barriers);

  // convert the shape of the agent into the type of the executor's shape
  using executor_shape_type = executor_shape_t<executor_type>;
  executor_shape_type executor_shape = detail::shape_cast<executor_shape_type>(agent_shape);

  // create the function that will marshal parameters received from bulk_invoke(executor) and execute the agent
  auto lambda = execute_agent_functor<executor_type,agent_traits,Function,UserArgIndices...>{param, agent_shape, executor_shape, f};

  return detail::bulk_async_with_executor(
    policy.executor(),
    executor_shape,
    lambda,
    std::forward<Args>(args)...,
    agency::share_at_scope_from_factory<SharedArgIndices>(agency::get<SharedArgIndices>(tuple_of_factories))...
  );
}


} // end detail
} // end agency

