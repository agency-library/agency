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
#include <agency/detail/execution_policy_traits.hpp>

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
  auto operator()(const Index& idx, Types&... packaged_shared_params) const
    -> decltype(
         __tu::tuple_apply(
           g,
           __tu::tuple_prepend_invoke(
             agency::detail::unpack_shared_parameters_from_executor(packaged_shared_params...),
             idx,
             agency::detail::forwarder{})
         )
       )
  {
    auto shared_params = agency::detail::unpack_shared_parameters_from_executor(packaged_shared_params...);

    // XXX the following is the moral equivalent of:
    // g(idx, shared_params...);

    // create one big tuple of the arguments so we can just call tuple_apply
    auto idx_and_shared_params = __tu::tuple_prepend_invoke(shared_params, idx, agency::detail::forwarder{});

    return __tu::tuple_apply(g, idx_and_shared_params);
  }
};

template<class Function>
__AGENCY_ANNOTATION
unpack_shared_parameters_from_executor_and_invoke<Function> make_unpack_shared_parameters_from_executor_and_invoke(Function f)
{
  return unpack_shared_parameters_from_executor_and_invoke<Function>{f};
}


template<class T>
struct decay_parameter : std::decay<T> {};

template<class T>
using decay_parameter_t = typename decay_parameter<T>::type;


template<size_t level, class T, class... Args>
struct decay_parameter<shared_parameter<level,T,Args...>>
{
  // shared_parameters are passed by reference
  using type = T&;
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


template<class Executor, class Function, class Tuple, size_t... TupleIndices>
auto bulk_invoke_executor_impl(Executor& exec, Function f, typename executor_traits<Executor>::shape_type shape, Tuple&& factory_tuple, detail::index_sequence<TupleIndices...>)
  -> decltype(
       executor_traits<Executor>::execute(exec, f, shape, detail::get<TupleIndices>(std::forward<Tuple>(factory_tuple))...)
     )
{
  return executor_traits<Executor>::execute(exec, f, shape, detail::get<TupleIndices>(std::forward<Tuple>(factory_tuple))...);
}

template<class Executor, class Function, class Tuple, size_t... TupleIndices>
typename executor_traits<Executor>::template future<void>
  bulk_async_executor_impl(Executor& exec, Function f, typename executor_traits<Executor>::shape_type shape, Tuple&& factory_tuple, detail::index_sequence<TupleIndices...>)
{
  return executor_traits<Executor>::async_execute(exec, f, shape, detail::get<TupleIndices>(std::forward<Tuple>(factory_tuple))...);
}


// computes the result type of bulk_invoke(executor)
template<class Executor, class Function, class... Args>
struct bulk_invoke_executor_result
{
  using function_result = result_of_t<
    Function(executor_index_t<Executor>, decay_parameter_t<Args>...)
  >;

  using type = executor_result_t<Executor, function_result>;
};

template<class Executor, class Function, class... Args>
using bulk_invoke_executor_result_t = typename bulk_invoke_executor_result<Executor,Function,Args...>::type;


template<bool enable, class Executor, class Function, class... Args>
struct enable_if_bulk_invoke_executor_impl {};

template<class Executor, class Function, class... Args>
struct enable_if_bulk_invoke_executor_impl<
         true, Executor, Function, Args...
       >
  : enable_if_call_possible<
      bulk_invoke_executor_result_t<Executor,Function,Args...>,
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


} // end detail


template<class Executor, class Function, class... Args>
typename detail::enable_if_bulk_invoke_executor<Executor, Function, Args...>::type
  bulk_invoke(Executor& exec, typename executor_traits<typename std::decay<Executor>::type>::shape_type shape, Function f, Args&&... args)
{
  // the _1 is for the executor idx parameter, which is the first parameter passed to f
  auto g = detail::bind_unshared_parameters(f, detail::placeholders::_1, std::forward<Args>(args)...);

  // make a tuple of the shared args
  auto shared_arg_tuple = detail::forward_shared_parameters_as_tuple(std::forward<Args>(args)...);

  using traits = executor_traits<Executor>;

  // package up the shared parameters for the executor
  const size_t executor_depth = detail::execution_depth<
    typename traits::execution_category
  >::value;

  // create a tuple of factories to use for shared parameters for the executor
  auto factory_tuple = agency::detail::make_shared_parameter_factory_tuple<executor_depth>(shared_arg_tuple);

  // unpack shared parameters we receive from the executor
  auto h = detail::make_unpack_shared_parameters_from_executor_and_invoke(g);

  return detail::bulk_invoke_executor_impl(exec, h, shape, factory_tuple, detail::make_index_sequence<executor_depth>());
}


template<class Executor, class Function, class... Args>
typename detail::enable_if_bulk_async_executor<Executor, Function, Args...>::type
  bulk_async(Executor& exec, typename executor_traits<typename std::decay<Executor>::type>::shape_type shape, Function f, Args&&... args)
{
  // the _1 is for the executor idx parameter, which is the first parameter passed to f
  auto g = detail::bind_unshared_parameters(f, detail::placeholders::_1, std::forward<Args>(args)...);

  // make a tuple of the shared args
  auto shared_arg_tuple = detail::forward_shared_parameters_as_tuple(std::forward<Args>(args)...);

  using traits = executor_traits<Executor>;

  // package up the shared parameters for the executor
  const size_t executor_depth = detail::execution_depth<
    typename traits::execution_category
  >::value;

  // create a tuple of factories to use for shared parameters for the executor
  auto factory_tuple = agency::detail::make_shared_parameter_factory_tuple<executor_depth>(shared_arg_tuple);

  // unpack shared parameters we receive from the executor
  auto h = detail::make_unpack_shared_parameters_from_executor_and_invoke(g);

  return detail::bulk_async_executor_impl(exec, h, shape, factory_tuple, detail::make_index_sequence<executor_depth>());
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

  template<class OtherFunction, class Tuple, size_t... Indices>
  static result_of_t<OtherFunction(agent_type&)>
    unpack_shared_params_and_execute(OtherFunction f, const agent_index_type& index, const agent_param_type& param, Tuple&& shared_params, detail::index_sequence<Indices...>)
  {
    return AgentTraits::execute(f, index, param, detail::get<Indices>(std::forward<Tuple>(shared_params))...);
  }

  template<class... Args>
  result_of_t<Function(agent_type&, pack_element_t<UserArgIndices, Args&&...>...)>
    operator()(const executor_index_type& executor_idx, Args&&... args)
  {
    // collect all parameters into a tuple of references
    auto args_tuple = detail::forward_as_tuple(std::forward<Args>(args)...);

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
      return f_(self, detail::get<UserArgIndices>(user_args)...);
    };

    constexpr size_t num_shared_args = std::tuple_size<decltype(agent_shared_args)>::value;
    return this->unpack_shared_params_and_execute(invoke_f, agent_idx, agent_param_, agent_shared_args, detail::make_index_sequence<num_shared_args>());
  }
};


template<class ExecutionPolicy, class Function, class... Args>
struct bulk_invoke_execution_policy_result
{
  using function_result = result_of_t<
    Function(execution_policy_agent_t<ExecutionPolicy>&, decay_parameter_t<Args>...)
  >;

  using type = executor_result_t<execution_policy_executor_t<ExecutionPolicy>, function_result>;
};

template<class ExecutionPolicy, class Function, class... Args>
using bulk_invoke_execution_policy_result_t = typename bulk_invoke_execution_policy_result<ExecutionPolicy,Function,Args...>::type;


template<size_t... UserArgIndices, size_t... SharedArgIndices, class ExecutionPolicy, class Function, class... Args>
bulk_invoke_execution_policy_result_t<
  ExecutionPolicy, Function, Args...
>
  bulk_invoke_execution_policy_impl(index_sequence<UserArgIndices...>,
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

  return bulk_invoke(policy.executor(), executor_shape, lambda, std::forward<Args>(args)..., share<SharedArgIndices>(detail::get<SharedArgIndices>(agent_shared_param_tuple))...);
}


template<size_t... UserArgIndices, size_t... SharedArgIndices, class ExecutionPolicy, class Function, class... Args>
policy_future_t<ExecutionPolicy,void>
  bulk_async_execution_policy_impl(index_sequence<UserArgIndices...>,
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

  return bulk_async(policy.executor(), executor_shape, lambda, std::forward<Args>(args)..., share<SharedArgIndices>(detail::get<SharedArgIndices>(agent_shared_param_tuple))...);
}


template<bool enable, class ExecutionPolicy, class Function, class... Args>
struct enable_if_bulk_invoke_execution_policy_impl {};

template<class ExecutionPolicy, class Function, class... Args>
struct enable_if_bulk_invoke_execution_policy_impl<
         true, ExecutionPolicy, Function, Args...
       >
  : enable_if_call_possible<
      bulk_invoke_execution_policy_result_t<ExecutionPolicy,Function,Args...>,
      Function,
      execution_policy_agent_t<ExecutionPolicy>&,
      decay_parameter_t<Args>...
    >
{};


template<class ExecutionPolicy, class Function, class... Args>
struct enable_if_bulk_invoke_execution_policy
  : enable_if_bulk_invoke_execution_policy_impl<
      has_execution_agent_type<
        typename std::decay<ExecutionPolicy>::type
      >::value,
      typename std::decay<ExecutionPolicy>::type,
      Function,
      Args...
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
      has_execution_agent_type<
        typename std::decay<ExecutionPolicy>::type
      >::value,
      typename std::decay<ExecutionPolicy>::type,
      Function,
      Args...
    >
{};


} // end detail


template<class ExecutionPolicy, class Function, class... Args>
typename detail::enable_if_bulk_invoke_execution_policy<
  ExecutionPolicy, Function, Args...
>::type
  bulk_invoke(ExecutionPolicy&& policy, Function f, Args&&... args)
{
  using agent_traits = execution_agent_traits<typename std::decay<ExecutionPolicy>::type::execution_agent_type>;
  const size_t num_shared_params = detail::execution_depth<typename agent_traits::execution_category>::value;

  return detail::bulk_invoke_execution_policy_impl(detail::index_sequence_for<Args...>(), detail::make_index_sequence<num_shared_params>(), policy, f, std::forward<Args>(args)...);
}


template<class ExecutionPolicy, class Function, class... Args>
typename detail::enable_if_bulk_async_execution_policy<
  ExecutionPolicy, Function, Args...
>::type
  bulk_async(ExecutionPolicy&& policy, Function&& f, Args&&... args)
{
  using agent_traits = execution_agent_traits<typename std::decay<ExecutionPolicy>::type::execution_agent_type>;
  const size_t num_shared_params = detail::execution_depth<typename agent_traits::execution_category>::value;

  using result_type = detail::policy_future_t<detail::decay_t<ExecutionPolicy>,void>;

  return detail::bulk_async_execution_policy_impl(detail::index_sequence_for<Args...>(), detail::make_index_sequence<num_shared_params>(), policy, f, std::forward<Args>(args)...);
}


} // end agency

