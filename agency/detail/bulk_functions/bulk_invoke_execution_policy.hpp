#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/bulk_functions/executor_functions/bulk_invoke_executor.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/bulk_functions/decay_parameter.hpp>
#include <agency/detail/bulk_functions/execute_agent_functor.hpp>
#include <agency/detail/bulk_functions/scope_result.hpp>
#include <agency/detail/bulk_functions/single_result.hpp>
#include <agency/detail/execution_policy_traits.hpp>
#include <agency/execution_agent.hpp>
#include <agency/executor/executor_traits.hpp>
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
    detail::is_scope_result<user_function_result>::value,
    detail::scope_result_to_bulk_invoke_result<user_function_result, execution_policy_executor_t<ExecutionPolicy>>,
    executor_result<execution_policy_executor_t<ExecutionPolicy>, user_function_result>
  >::type;
};


template<class ExecutionPolicy, class Function, class... Args>
using bulk_invoke_execution_policy_result_t = typename bulk_invoke_execution_policy_result<ExecutionPolicy,Function,Args...>::type;


template<size_t... UserArgIndices, size_t... SharedArgIndices, class ExecutionPolicy, class Function, class... Args>
bulk_invoke_execution_policy_result_t<
  ExecutionPolicy, Function, Args...
>
  bulk_invoke_execution_policy(index_sequence<UserArgIndices...>,
                               index_sequence<SharedArgIndices...>,
                               ExecutionPolicy& policy, Function f, Args&&... args)
{
  using agent_type = typename ExecutionPolicy::execution_agent_type;
  using agent_traits = execution_agent_traits<agent_type>;
  using execution_category = typename agent_traits::execution_category;

  // get the parameters of the agent
  auto param = policy.param();
  auto agent_shape = agent_traits::domain(param).shape();

  auto agent_shared_param_tuple = agent_traits::make_shared_param_tuple(param);

  using executor_type = typename ExecutionPolicy::executor_type;
  using executor_traits = agency::executor_traits<executor_type>;

  // convert the shape of the agent into the type of the executor's shape
  using executor_shape_type = typename executor_traits::shape_type;
  executor_shape_type executor_shape = detail::shape_cast<executor_shape_type>(agent_shape);

  // create the function that will marshal parameters received from bulk_invoke(executor) and execute the agent
  auto lambda = execute_agent_functor<executor_traits,agent_traits,Function,UserArgIndices...>{param, agent_shape, executor_shape, f};

  return detail::bulk_invoke_executor(policy.executor(), executor_shape, lambda, std::forward<Args>(args)..., agency::share_at_scope<SharedArgIndices>(detail::get<SharedArgIndices>(agent_shared_param_tuple))...);
}


} // end detail
} // end agency

