#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/type_list.hpp>
#include <agency/detail/control_structures/executor_functions/bulk_then_with_executor.hpp>
#include <agency/detail/control_structures/decay_parameter.hpp>
#include <agency/detail/control_structures/single_result.hpp>
#include <agency/detail/control_structures/shared_parameter.hpp>
#include <agency/detail/control_structures/tuple_of_agent_shared_parameter_factories.hpp>
#include <agency/detail/control_structures/bulk_invoke_execution_policy.hpp>
#include <agency/detail/is_call_possible.hpp>
#include <agency/execution/execution_agent.hpp>
#include <agency/execution/executor/executor_traits/executor_shape.hpp>
#include <agency/execution/executor/executor_traits/detail/executor_barrier_types_as_scoped_in_place_type.hpp>
#include <agency/execution/execution_policy/execution_policy_traits.hpp>
#include <agency/tuple.hpp>
#include <utility>

namespace agency
{
namespace detail
{


template<class Executor, class AgentTraits, class Function, class Future, size_t... UserArgIndices>
struct then_execute_agent_functor
{
  // XXX should just make the future's result_type a parameter of this functor and try to use it SFINAE the operator()s below
  using agent_type        = typename AgentTraits::execution_agent_type;
  using agent_param_type  = typename AgentTraits::param_type;
  using agent_domain_type = typename AgentTraits::domain_type;
  using agent_shape_type  = decltype(std::declval<agent_domain_type>().shape());

  using executor_shape_type = executor_shape_t<Executor>;

  agent_param_type    agent_param_;
  agent_shape_type    agent_shape_;
  executor_shape_type executor_shape_;
  Function            f_;

  using agent_index_type    = typename AgentTraits::index_type;
  using executor_index_type = executor_index_t<Executor>;

  template<class OtherFunction, class Tuple, size_t... Indices>
  __AGENCY_ANNOTATION
  static result_of_t<OtherFunction(agent_type&)>
    unpack_shared_params_and_execute(OtherFunction f, const agent_index_type& index, const agent_param_type& param, Tuple&& shared_params, detail::index_sequence<Indices...>)
  {
    return AgentTraits::execute(f, index, param, agency::get<Indices>(std::forward<Tuple>(shared_params))...);
  }

  // this overload of operator() handles the case where the Future given to then_execute() is non-void
  template<class PastArg, class... Args,
           class Future1 = Future,
           class = typename std::enable_if<
             is_non_void_future<Future1>::value
           >::type>
  __AGENCY_ANNOTATION
  result_of_continuation_t<Function, agent_type&, Future, pack_element_t<UserArgIndices, Args...>...>
    operator()(const executor_index_type& executor_idx, PastArg& past_arg, Args&&... args)
  {
    // collect all parameters into a tuple of references
    auto args_tuple = agency::forward_as_tuple(std::forward<Args>(args)...);

    // split the parameters into user parameters and agent parameters
    auto user_args         = detail::tuple_take_view<sizeof...(UserArgIndices)>(args_tuple);
    auto agent_shared_args = detail::tuple_drop_view<sizeof...(UserArgIndices)>(args_tuple);

    // turn the executor index into an agent index
    auto agent_idx = detail::index_cast<agent_index_type>(executor_idx, executor_shape_, agent_shape_);

    // AgentTraits::execute expects a function whose only parameter is agent_type
    // so we have to wrap f_ into a function of one parameter
    auto invoke_f = [&past_arg,&user_args,this](agent_type& self)
    {
      // invoke f by passing the agent, then the past_arg, then the user's parameters
      return f_(self, past_arg, agency::get<UserArgIndices>(user_args)...);
    };

    return this->unpack_shared_params_and_execute(invoke_f, agent_idx, agent_param_, agent_shared_args, detail::make_tuple_indices(agent_shared_args));
  }

  // this overload of operator() handles the case where the Future given to then_execute() is void
  // it is identical to the one above except that past_arg does not exist
  template<class... Args,
           class Future1 = Future,
           class = typename std::enable_if<
             is_void_future<Future1>::value
           >::type>
  __AGENCY_ANNOTATION
  result_of_continuation_t<Function, agent_type&, Future, pack_element_t<UserArgIndices, Args...>...>
    operator()(const executor_index_type& executor_idx, Args&&... args)
  {
    // collect all parameters into a tuple of references
    auto args_tuple = agency::forward_as_tuple(std::forward<Args>(args)...);

    // split the parameters into user parameters and agent parameters
    auto user_args         = detail::tuple_take_view<sizeof...(UserArgIndices)>(args_tuple);
    auto agent_shared_args = detail::tuple_drop_view<sizeof...(UserArgIndices)>(args_tuple);

    // turn the executor index into an agent index
    auto agent_idx = detail::index_cast<agent_index_type>(executor_idx, executor_shape_, agent_shape_);

    // AgentTraits::execute expects a function whose only parameter is agent_type
    // so we have to wrap f_ into a function of one parameter
    auto invoke_f = [&user_args,this](agent_type& self)
    {
      // invoke f by passing the agent, then the user's parameters
      return f_(self, agency::get<UserArgIndices>(user_args)...);
    };

    return this->unpack_shared_params_and_execute(invoke_f, agent_idx, agent_param_, agent_shared_args, detail::make_tuple_indices(agent_shared_args));
  }
};


template<class ExecutionPolicy, class Function, class Future, class... Args>
struct bulk_then_execution_policy_result
{
  // figure out the Future's result_type
  using future_result_type = future_result_t<Future>;

  // avoid passing Future to bulk_invoke_execution_policy_result when it is a void Future
  using bulk_invoke_result_type = typename detail::lazy_conditional<
    std::is_void<future_result_type>::value,
    bulk_invoke_execution_policy_result<ExecutionPolicy,Function,Args...>,
    bulk_invoke_execution_policy_result<ExecutionPolicy,Function,Future,Args...>
  >::type;

  using type = execution_policy_future_t<
    ExecutionPolicy,
    bulk_invoke_result_type
  >;
};

template<class ExecutionPolicy, class Function, class Future, class... Args>
using bulk_then_execution_policy_result_t = typename bulk_then_execution_policy_result<ExecutionPolicy,Function,Future,Args...>::type;


template<size_t... UserArgIndices, size_t... SharedArgIndices, class ExecutionPolicy, class Function, class Future, class... Args>
__AGENCY_ANNOTATION
bulk_then_execution_policy_result_t<
  ExecutionPolicy, Function, Future, Args...
>
  bulk_then_execution_policy(index_sequence<UserArgIndices...>,
                             index_sequence<SharedArgIndices...>,
                             ExecutionPolicy& policy, Function f, Future& fut, Args&&... args)
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
  auto lambda = then_execute_agent_functor<executor_type,agent_traits,Function,Future,UserArgIndices...>{param, agent_shape, executor_shape, f};

  return detail::bulk_then_with_executor(
    policy.executor(),
    executor_shape,
    lambda,
    fut,
    std::forward<Args>(args)...,
    agency::share_at_scope_from_factory<SharedArgIndices>(agency::get<SharedArgIndices>(tuple_of_factories))...
  );
}


template<class ExecutionPolicy, class Function, class Future, class... Args>
struct is_bulk_then_possible_via_execution_policy_impl
{
  template<class ExecutionPolicy1, class Function1, class Future1, class... Args1,
           class = typename std::enable_if<
             has_execution_agent_type<ExecutionPolicy1>::value
           >::type,
           class = typename std::enable_if<
             is_non_void_future<Future1>::value
           >::type,
           class = typename enable_if_call_possible<
             void, Function1, execution_policy_agent_t<ExecutionPolicy1>&, decay_parameter_t<Future1>, decay_parameter_t<Args1>...
           >::type
          >
  static std::true_type test_non_void(int);

  template<class...>
  static std::false_type test_non_void(...);

  template<class ExecutionPolicy1, class Function1, class Future1, class... Args1,
           class = typename std::enable_if<
             has_execution_agent_type<ExecutionPolicy1>::value
           >::type,
           class = typename std::enable_if<
             is_void_future<Future1>::value
           >::type,
           class = typename enable_if_call_possible<
             void, Function1, execution_policy_agent_t<ExecutionPolicy1>&, decay_parameter_t<Args1>...
           >::type
          >
  static std::true_type test_void(int);
  
  template<class...>
  static std::false_type test_void(...);

  // there are two tests: one applies when Future is a void future
  using test_void_result = decltype(test_void<ExecutionPolicy,Function,Future,Args...>(0));

  // ther other applies when Future is a non-void future
  using test_non_void_result = decltype(test_non_void<ExecutionPolicy,Function,Future,Args...>(0));

  // if either test passed, then the result is true
  using type = detail::disjunction<test_void_result,test_non_void_result>;
};

template<class ExecutionPolicy, class Function, class Future, class... Args>
using is_bulk_then_possible_via_execution_policy = typename is_bulk_then_possible_via_execution_policy_impl<ExecutionPolicy,Function,Future,Args...>::type;


} // end detail
} // end agency

