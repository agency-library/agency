#pragma once

#include <agency/detail/config.hpp>
#include <agency/executor_traits.hpp>
#include <agency/execution_agent.hpp>
#include <agency/functional.hpp>
#include <agency/detail/shared_parameter.hpp>
#include <agency/detail/is_call_possible.hpp>
#include <agency/detail/shape_cast.hpp>
#include <agency/detail/index_cast.hpp>
#include <agency/detail/tuple.hpp>

namespace agency
{


template<class ExecutionPolicy>
struct is_execution_policy;


namespace detail
{


template<class Function>
struct unpack_shared_parameters_from_executor_and_invoke
{
  mutable Function g;

  template<class Index, class... Types>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Types&... packaged_shared_params) const
  {
    auto shared_params = agency::detail::unpack_shared_parameters_from_executor(packaged_shared_params...);

    // XXX the following is the moral equivalent of:
    // g(idx, shared_params...);

    // create one big tuple of the arguments so we can just call tuple_apply
    auto idx_and_shared_params = __tu::tuple_prepend_invoke(shared_params, idx, agency::detail::forwarder{});

    __tu::tuple_apply(g, idx_and_shared_params);
  }
};


template<class BulkCall, class Executor, class Function, class Tuple, size_t... TupleIndices>
typename BulkCall::result_type
  bulk_call_executor_impl(BulkCall bulk_call, Executor& exec, Function f, typename executor_traits<Executor>::shape_type shape, Tuple&& shared_init_tuple, detail::index_sequence<TupleIndices...>)
{
  return bulk_call(exec, unpack_shared_parameters_from_executor_and_invoke<Function>{f}, shape, detail::get<TupleIndices>(std::forward<Tuple>(shared_init_tuple))...);
}


// since almost all the code is shared between bulk_invoke_executor & bulk_async,
// we collapse it all into one function parameterized by the bulk call in question
template<class BulkCall, class Executor, class Function, class... Args>
typename BulkCall::result_type
  bulk_call_executor(BulkCall bulk_call, Executor& exec, typename executor_traits<Executor>::shape_type shape, Function f, Args&&... args)
{
  // the _1 is for the executor idx parameter, which is the first parameter passed to f
  auto g = bind_unshared_parameters(f, placeholders::_1, std::forward<Args>(args)...);

  // make a tuple of the shared args
  auto shared_arg_tuple = forward_shared_parameters_as_tuple(std::forward<Args>(args)...);

  using traits = executor_traits<Executor>;

  // package up the shared parameters for the executor
  const size_t executor_depth = detail::execution_depth<
    typename traits::execution_category
  >::value;

  // construct shared initializers and package them for the executor
  auto shared_init = agency::detail::make_shared_parameter_package_for_executor<executor_depth>(shared_arg_tuple);

  return bulk_call_executor_impl(bulk_call, exec, g, shape, std::move(shared_init), detail::make_index_sequence<executor_depth>());

  // XXX upon c++14
  //return bulk_call(exec, [=](const auto& idx, auto& packaged_shared_params)
  //{
  //  auto shared_params = agency::detail::unpack_shared_parameters_from_executor(packaged_shared_params);

  //  // XXX the following is the moral equivalent of:
  //  // g(idx, shared_params...);

  //  // create one big tuple of the arguments so we can just call tuple_apply
  //  auto idx_and_shared_params = __tu::tuple_prepend_invoke(shared_params, idx, agency::detail::forwarder{});

  //  __tu::tuple_apply(g, idx_and_shared_params);
  //},
  //shape,
  //std::move(shared_init));
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


template<class T>
struct decay_parameter : decay_construct_result<T> {};

template<class T>
using decay_parameter_t = typename decay_parameter<T>::type;


template<size_t level, class T, class... Args>
struct decay_parameter<shared_parameter<level,T,Args...>>
{
  // shared_parameters are passed by reference
  using type = T&;
};


template<bool enable, class Executor, class Function, class... Args>
struct enable_if_bulk_invoke_executor_impl {};

template<class Executor, class Function, class... Args>
struct enable_if_bulk_invoke_executor_impl<
         true, Executor, Function, Args...
       >
  : enable_if_call_possible<
      void,
      Function,
      executor_index_t<Executor>,
      decay_parameter_t<Args>...
    >
{};

template<class Executor, class Function, class... Args>
struct enable_if_bulk_invoke_executor
  : enable_if_bulk_invoke_executor_impl<
      is_executor<Executor>::value, Executor, Function, Args...
    >
{};


template<class Executor, class Function, class... Args>
typename enable_if_bulk_invoke_executor<Executor, Function, Args...>::type
  bulk_invoke_executor(Executor& exec, typename executor_traits<typename std::decay<Executor>::type>::shape_type shape, Function f, Args&&... args)
{
  call_execute<Executor> caller;
  return bulk_call_executor(caller, exec, shape, f, std::forward<Args>(args)...);
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


template<bool enable, class Executor, class Function, class... Args>
struct enable_if_bulk_async_executor_impl {};

template<class Executor, class Function, class... Args>
struct enable_if_bulk_async_executor_impl<
         true, Executor, Function, Args...
       >
  : enable_if_call_possible<
      executor_future_t<Executor,void>,
      Function,
      executor_index_t<Executor>,
      decay_parameter_t<Args>...
    >
{};

template<class Executor, class Function, class... Args>
struct enable_if_bulk_async_executor
  : enable_if_bulk_async_executor_impl<
      is_executor<Executor>::value, Executor, Function, Args...
    >
{};


} // end detail


template<class Executor, class Function, class... Args>
typename detail::enable_if_bulk_async_executor<Executor, Function, Args...>::type
  bulk_async(Executor& exec, typename executor_traits<typename std::decay<Executor>::type>::shape_type shape, Function f, Args&&... args)
{
  using result_type = detail::executor_future_t<Executor,void>;

  detail::call_async_execute<Executor,result_type> caller;
  return detail::bulk_call_executor(caller, exec, shape, f, std::forward<Args>(args)...);
}


namespace detail
{


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
  void operator()(const executor_index_type& executor_idx, Args&&... args)
  {
    // collect all parameters into a tuple of references
    auto args_tuple = detail::forward_as_tuple(std::forward<Args>(args)...);

    // split the parameters into user parameters and agent parameters
    auto user_args         = detail::tuple_take_view<sizeof...(UserArgIndices)>(args_tuple);
    auto agent_shared_args = detail::tuple_drop_view<sizeof...(UserArgIndices)>(args_tuple);

    // turn the executor index into an agent index
    using agent_index_type = typename AgentTraits::index_type;
    auto agent_idx = detail::index_cast<agent_index_type>(executor_idx, executor_shape_, agent_shape_);

    // AgentTraits::execute expects a function whose only parameter is agent_type
    // so we have to wrap f_ into a function of one parameter
    auto invoke_f = [&user_args,this](agent_type& self)
    {
      // invoke f by passing the agent, then the user's parameters
      f_(self, detail::get<UserArgIndices>(user_args)...);
    };

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
                             index_sequence<UserArgIndices...>,
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

  return bulk_call(policy.executor(), executor_shape, lambda, std::forward<Args>(args)..., share<SharedArgIndices>(detail::get<SharedArgIndices>(agent_shared_param_tuple))...);
}


// this functor takes its arguments and forwards them to bulk_invoke_executor
struct call_bulk_invoke_executor
{
  using result_type = void;

  template<class... Args>
  void operator()(Args&&... args)
  {
    detail::bulk_invoke_executor(std::forward<Args>(args)...);
  }
};


// this functor takes its arguments and calls bulk_async via adl
template<class Result>
struct call_bulk_async_adl
{
  using result_type = Result;

  template<class... Args>
  result_type operator()(Args&&... args)
  {
    using agency::bulk_async;
    return bulk_async(std::forward<Args>(args)...);
  }
};


__DEFINE_HAS_NESTED_TYPE(has_executor_type, executor_type);

template<class ExecutionPolicy, class T, class Enable = void>
struct policy_future {};

// gets the type of future bulk_async(policy) returns
template<class ExecutionPolicy, class T>
struct policy_future<ExecutionPolicy,T,typename std::enable_if<has_executor_type<ExecutionPolicy>::value>::type>
{
  using type = executor_future_t<typename ExecutionPolicy::executor_type, T>;
};

template<class ExecutionPolicy, class T>
using policy_future_t = typename policy_future<ExecutionPolicy,T>::type;


__DEFINE_HAS_NESTED_TYPE(has_execution_agent_type, execution_agent_type);

template<class ExecutionPolicy, class Enable = void>
struct execution_policy_agent {};

template<class ExecutionPolicy>
struct execution_policy_agent<ExecutionPolicy,typename std::enable_if<has_execution_agent_type<ExecutionPolicy>::value>::type>
{
  using type = typename ExecutionPolicy::execution_agent_type;
};


template<class ExecutionPolicy>
using execution_policy_agent_t = typename execution_policy_agent<ExecutionPolicy>::type;


template<bool enable, class ExecutionPolicy, class Function, class... Args>
struct enable_if_bulk_invoke_execution_policy_impl {};

template<class ExecutionPolicy, class Function, class... Args>
struct enable_if_bulk_invoke_execution_policy_impl<
         true, ExecutionPolicy, Function, Args...
       >
  : enable_if_call_possible<
      void,
      Function,
      execution_policy_agent_t<ExecutionPolicy>&,
      decay_parameter_t<Args>...
    >
{};


template<class ExecutionPolicy, class Function, class... Args>
struct enable_if_bulk_invoke_execution_policy
  : enable_if_bulk_invoke_execution_policy_impl<
      has_execution_agent_type<ExecutionPolicy>::value, ExecutionPolicy, Function, Args...
    >
{};


template<bool enable, class ExecutionPolicy, class Function, class... Args>
struct enable_if_bulk_async_execution_policy_impl {};

template<class ExecutionPolicy, class Function, class... Args>
struct enable_if_bulk_async_execution_policy_impl<
         true, ExecutionPolicy, Function, Args...
       >
  : enable_if_call_possible<
      policy_future_t<ExecutionPolicy,void>,
      Function,
      execution_policy_agent_t<ExecutionPolicy>&,
      decay_parameter_t<Args>...
    >
{};


template<class ExecutionPolicy, class Function, class... Args>
struct enable_if_bulk_async_execution_policy
  : enable_if_bulk_async_execution_policy_impl<
      has_execution_agent_type<ExecutionPolicy>::value, ExecutionPolicy, Function, Args...
    >
{};


} // end detail


template<class ExecutionPolicy, class Function, class... Args>
typename detail::enable_if_bulk_invoke_execution_policy<
  typename std::decay<ExecutionPolicy>::type, Function, Args...
>::type
  bulk_invoke(ExecutionPolicy&& policy, Function f, Args&&... args)
{
  using agent_traits = execution_agent_traits<typename std::decay<ExecutionPolicy>::type::execution_agent_type>;
  const size_t num_shared_params = detail::execution_depth<typename agent_traits::execution_category>::value;

  detail::call_bulk_invoke_executor invoker;
  detail::bulk_call_execution_policy(invoker, detail::index_sequence_for<Args...>(), detail::make_index_sequence<num_shared_params>(), policy, f, std::forward<Args>(args)...);
}


template<class ExecutionPolicy, class Function, class... Args>
typename detail::enable_if_bulk_async_execution_policy<
  typename std::decay<ExecutionPolicy>::type, Function, Args...
>::type
  bulk_async(ExecutionPolicy&& policy, Function&& f, Args&&... args)
{
  using agent_traits = execution_agent_traits<typename std::decay<ExecutionPolicy>::type::execution_agent_type>;
  const size_t num_shared_params = detail::execution_depth<typename agent_traits::execution_category>::value;

  using result_type = detail::policy_future_t<detail::decay_t<ExecutionPolicy>,void>;

  detail::call_bulk_async_adl<result_type> asyncer;
  return detail::bulk_call_execution_policy(asyncer, detail::index_sequence_for<Args...>(), detail::make_index_sequence<num_shared_params>(), policy, f, std::forward<Args>(args)...);
}


} // end agency

