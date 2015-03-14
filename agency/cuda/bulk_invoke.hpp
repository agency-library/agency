#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/shared_parameter.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/executor_traits.hpp>
#include <agency/execution_agent.hpp>
#include <agency/cuda/detail/bind.hpp>
#include <utility>
#include <tuple>
#include <type_traits>


namespace agency
{
namespace cuda
{
namespace detail
{

  
template<class Function, class Indices>
struct unpack_shared_parameters_from_executor_and_invoke;


template<class Function, size_t... I>
struct unpack_shared_parameters_from_executor_and_invoke<Function,agency::detail::index_sequence<I...>>
{
  mutable Function g;

  template<class Index, class SharedParams>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, SharedParams& packaged_shared_params) const
  {
    auto shared_params = agency::detail::unpack_shared_parameters_from_executor(packaged_shared_params);

    g(idx, agency::detail::get<I>(shared_params)...);
  }
};


// since almost all the code is shared between bulk_invoke_executor & bulk_async_executor,
// we collapse it all into one function parameterized by the bulk call in question
template<class BulkCall, class Executor, class Function, class... Args>
typename BulkCall::result_type
  bulk_call_executor(BulkCall bulk_call, Executor& exec, Function f, typename agency::executor_traits<typename std::decay<Executor>::type>::shape_type shape, Args&&... args)
{
  // the _1 is for the executor idx parameter, which is the first parameter passed to f
  auto g = detail::bind_unshared_parameters(f, placeholders::_1, std::forward<Args>(args)...);

  // make a tuple of the shared args
  auto shared_arg_tuple = agency::detail::forward_shared_parameters_as_tuple(std::forward<Args>(args)...);

  using traits = executor_traits<Executor>;

  // package up the shared parameters for the executor
  const size_t executor_depth = agency::detail::execution_depth<
    typename traits::execution_category
  >::value;

  // construct shared arguments and package them for the executor
  auto packaged_shared_parameter_tuple = agency::detail::make_shared_parameter_package_for_executor<executor_depth>(shared_arg_tuple);

  auto functor = unpack_shared_parameters_from_executor_and_invoke<
    decltype(g),
    agency::detail::make_index_sequence<std::tuple_size<decltype(shared_arg_tuple)>::value>
  >{g};

  return bulk_call(exec, functor, shape, std::move(packaged_shared_parameter_tuple));
}


template<class Executor>
struct call_execute
{
  using result_type = void;

  template<class... Args>
  void operator()(Args&&... args)
  {
    executor_traits<Executor>::execute(std::forward<Args>(args)...);
  }
};


template<class Executor, class Function, class... Args>
typename agency::detail::enable_if_bulk_invoke_executor<
  Executor, Function, Args...
>::type
  bulk_invoke_executor(Executor& exec, Function f, typename executor_traits<typename std::decay<Executor>::type>::shape_type shape, Args&&... args)
{
  call_execute<Executor> caller;
  return detail::bulk_call_executor(caller, exec, f, shape, std::forward<Args>(args)...);
}


template<class Executor, class Result>
struct call_async_execute
{
  using result_type = Result;

  template<class... Args>
  result_type operator()(Args&&... args)
  {
    return executor_traits<Executor>::async_execute(std::forward<Args>(args)...);
  }
};


template<class Executor, class Function, class... Args>
typename agency::detail::enable_if_bulk_async_executor<
  Executor, Function, Args...
>::type
  bulk_async_executor(Executor& exec, Function f, typename executor_traits<typename std::decay<Executor>::type>::shape_type shape, Args&&... args)
{
  using result_type = agency::detail::executor_future<Executor,void>;

  call_async_execute<Executor,result_type> caller;
  return detail::bulk_call_executor(caller, exec, f, shape, std::forward<Args>(args)...);
}


template<class Function, class Tuple, size_t... Indices>
struct unpack_arguments_and_invoke_with_self
{
  Function& f;
  Tuple& args;

  template<class ExecutionAgent>
  __AGENCY_ANNOTATION
  void operator()(ExecutionAgent& self)
  {
    f(self, agency::detail::get<Indices>(args)...);
  }
};


template<class ExecutorTraits, class AgentTraits, class Function, size_t... UserArgIndices>
struct execute_agent_functor
{
  using agent_type        = typename AgentTraits::execution_agent_type;
  using agent_param_type  = typename AgentTraits::param_type;
  using agent_domain_type = typename AgentTraits::domain_type;
  using agent_shape_type  = decltype(std::declval<agent_domain_type>().shape());
  using agent_execution_category = typename AgentTraits::execution_category;

  using executor_shape_type = typename ExecutorTraits::shape_type;

  agent_param_type    agent_param_;
  agent_shape_type    agent_shape_;
  executor_shape_type executor_shape_;
  Function            f_;

  using agent_index_type    = typename AgentTraits::index_type;
  using executor_index_type = typename ExecutorTraits::index_type;

  template<class... Args>
  __AGENCY_ANNOTATION
  void operator()(const executor_index_type& executor_idx, Args&&... args)
  {
    // collect all parameters into a tuple of references
    auto args_tuple = agency::detail::forward_as_tuple(std::forward<Args>(args)...);

    // split the parameters into user parameters and agent parameters
    auto user_args         = agency::detail::tuple_take_view<sizeof...(UserArgIndices)>(args_tuple);
    auto agent_shared_args = agency::detail::tuple_drop_view<sizeof...(UserArgIndices)>(args_tuple);

    // turn the executor index into an agent index
    using agent_index_type = typename AgentTraits::index_type;
    agent_index_type agent_idx = agency::detail::index_cast<agent_index_type>(executor_idx, executor_shape_, agent_shape_);

    // AgentTraits::execute expects a function whose only parameter is agent_type
    // so we have to wrap f_ into a function of one parameter
    // XXX using this lambda instead of a named functor adds ~6 seconds of compilation time
    //auto invoke_f = [&user_args,this](agent_type& self)
    //{
    //  // invoke f by passing the agent, then the user's parameters
    //  f_(self, agency::detail::get<UserArgIndices>(user_args)...);
    //};
    unpack_arguments_and_invoke_with_self<Function,decltype(user_args),UserArgIndices...> invoke_f{f_, user_args};

    AgentTraits::execute(invoke_f, agent_idx, agent_param_, agent_shared_args);
  }
};


// this is the implementation of bulk_invoke(execution_policy) or bulk_async(execution_policy),
// depending on what BulkCall does
// the implementation of bulk_invoke & bulk_async are the same, modulo the type of result
// the idea is to lower the call onto the corresponding call to bulk_invoke_executor or bulk_async_executor
// by marshalling the shared arguments used to construct execution agents
template<class BulkCall, size_t... UserArgIndices, size_t... SharedArgIndices, class ExecutionPolicy, class Function, class... Args>
typename BulkCall::result_type
  bulk_call_execution_policy(BulkCall bulk_call,
                             agency::detail::index_sequence<UserArgIndices...>,
                             agency::detail::index_sequence<SharedArgIndices...>,
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
  using executor_traits = executor_traits<executor_type>;

  // convert the shape of the agent into the type of the executor's shape
  using executor_shape_type = typename executor_traits::shape_type;
  executor_shape_type executor_shape = agency::detail::shape_cast<executor_shape_type>(agent_shape);

  // create the function that will marshal parameters received from bulk_invoke(executor) and execute the agent
  auto invoke_me = execute_agent_functor<executor_traits,agent_traits,Function,UserArgIndices...>{param, agent_shape, executor_shape, f};

  return bulk_call(policy.executor(), invoke_me, executor_shape, std::forward<Args>(args)..., agency::share<SharedArgIndices>(agency::detail::get<SharedArgIndices>(agent_shared_param_tuple))...);
}


// this functor takes its arguments and forwards them to bulk_invoke_executor
struct call_bulk_invoke_executor
{
  using result_type = void;

  template<class... Args>
  void operator()(Args&&... args)
  {
    agency::cuda::detail::bulk_invoke_executor(std::forward<Args>(args)...);
  }
};


// this functor takes its arguments and forwards them to bulk_async_executor
template<class Result>
struct call_bulk_async_executor
{
  using result_type = Result;

  template<class... Args>
  result_type operator()(Args&&... args)
  {
    return agency::cuda::detail::bulk_async_executor(std::forward<Args>(args)...);
  }
};


} // end detail


template<class ExecutionPolicy, class Function, class... Args>
typename agency::detail::enable_if_bulk_invoke_execution_policy<
  ExecutionPolicy, Function, Args...
>::type
  bulk_invoke(ExecutionPolicy&& policy, Function f, Args&&... args)
{
  using agent_traits = execution_agent_traits<typename std::decay<ExecutionPolicy>::type::execution_agent_type>;
  const size_t num_shared_params = agency::detail::execution_depth<typename agent_traits::execution_category>::value;

  detail::call_bulk_invoke_executor invoker;
  detail::bulk_call_execution_policy(invoker, agency::detail::index_sequence_for<Args...>(), agency::detail::make_index_sequence<num_shared_params>(), policy, f, std::forward<Args>(args)...);
}


template<class ExecutionPolicy, class Function, class... Args>
typename agency::detail::enable_if_bulk_async_execution_policy<
  ExecutionPolicy, Function, Args...
>::type
  bulk_async(ExecutionPolicy&& policy, Function&& f, Args&&... args)
{
  using agent_traits = execution_agent_traits<typename std::decay<ExecutionPolicy>::type::execution_agent_type>;
  const size_t num_shared_params = agency::detail::execution_depth<typename agent_traits::execution_category>::value;

  using result_type = agency::detail::policy_future<agency::detail::decay_t<ExecutionPolicy>,void>;

  detail::call_bulk_async_executor<result_type> asyncer;
  return detail::bulk_call_execution_policy(asyncer, agency::detail::index_sequence_for<Args...>(), agency::detail::make_index_sequence<num_shared_params>(), policy, f, std::forward<Args>(args)...);
}


} // end cuda
} // end agency

