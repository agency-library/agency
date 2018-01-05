#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/control_structures/executor_functions/bulk_invoke_with_executor.hpp>
#include <agency/detail/control_structures/decay_parameter.hpp>
#include <agency/detail/control_structures/execute_agent_functor.hpp>
#include <agency/detail/control_structures/scope_result.hpp>
#include <agency/detail/control_structures/single_result.hpp>
#include <agency/detail/control_structures/shared_parameter.hpp>
#include <agency/detail/control_structures/tuple_of_agent_shared_parameter_factories.hpp>
#include <agency/execution/execution_agent.hpp>
#include <agency/execution/executor/executor_traits/executor_shape.hpp>
#include <agency/execution/executor/executor_traits/detail/executor_barrier_types_as_scoped_in_place_type.hpp>
#include <agency/execution/executor/detail/utility/executor_bulk_result_or_void.hpp>
#include <agency/execution/execution_policy/execution_policy_traits.hpp>
#include <agency/tuple.hpp>
#include <utility>

namespace agency
{
namespace detail
{


template<class ExecutionPolicy, class Function, class... Args>
struct bulk_invoke_execution_policy_result
{
  // first figure out what type the user function returns
  using user_function_result = result_of_t<
    Function(execution_policy_agent_t<ExecutionPolicy>&, decay_parameter_t<Args>...)
  >;

  // if the user function returns scope_result, then use scope_result_to_bulk_invoke_result to figure out what to return
  // else, the result is whatever executor_result<executor_type, function_result> thinks it is
  using type = typename detail::lazy_conditional<
    is_scope_result<user_function_result>::value,
    scope_result_to_bulk_invoke_result<user_function_result, execution_policy_executor_t<ExecutionPolicy>>,
    executor_bulk_result_or_void<execution_policy_executor_t<ExecutionPolicy>, user_function_result>
  >::type;
};


template<class ExecutionPolicy, class Function, class... Args>
using bulk_invoke_execution_policy_result_t = typename bulk_invoke_execution_policy_result<ExecutionPolicy,Function,Args...>::type;


template<size_t... UserArgIndices, size_t... SharedArgIndices, class ExecutionPolicy, class Function, class... Args>
__AGENCY_ANNOTATION
bulk_invoke_execution_policy_result_t<
  ExecutionPolicy, Function, Args...
>
  bulk_invoke_execution_policy(index_sequence<UserArgIndices...>,
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

  return detail::bulk_invoke_with_executor(
    policy.executor(),
    executor_shape,
    lambda,
    std::forward<Args>(args)...,
    agency::share_at_scope_from_factory<SharedArgIndices>(agency::get<SharedArgIndices>(tuple_of_factories))...
  );
}


} // end detail
} // end agency

