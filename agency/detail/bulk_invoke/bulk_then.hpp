#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/type_list.hpp>
#include <agency/future.hpp>

namespace agency
{
namespace detail
{


// this overload handles the general case where the user function returns a normal result
template<class Executor, class Function, class Factory, class Future, class Tuple, size_t... TupleIndices>
executor_future_t<Executor, typename std::result_of<Factory(executor_shape_t<Executor>)>::type>
  bulk_then_executor_impl(Executor& exec,
                          Function f,
                          Factory result_factory,
                          typename executor_traits<Executor>::shape_type shape,
                          Future& fut,
                          Tuple&& factory_tuple,
                          detail::index_sequence<TupleIndices...>)
{
  return executor_traits<Executor>::then_execute(exec, f, result_factory, shape, fut, detail::get<TupleIndices>(std::forward<Tuple>(factory_tuple))...);
}

// this overload handles the special case where the user function returns a scope_result
template<class Executor, class Function, size_t scope, class T, class Future, class Tuple, size_t... TupleIndices>
executor_future_t<Executor, typename detail::scope_result_container<scope,T,Executor>::result_type>
  bulk_then_executor_impl(Executor& exec,
                          Function f,
                          detail::executor_traits_detail::container_factory<detail::scope_result_container<scope,T,Executor>> result_factory,
                          typename executor_traits<Executor>::shape_type shape,
                          Future& fut,
                          Tuple&& factory_tuple,
                          detail::index_sequence<TupleIndices...>)
{
  auto intermediate_future = executor_traits<Executor>::then_execute(exec, f, result_factory, shape, fut, detail::get<TupleIndices>(std::forward<Tuple>(factory_tuple))...);

  using result_type = typename detail::scope_result_container<scope,T,Executor>::result_type;

  return executor_traits<Executor>::template future_cast<result_type>(exec, intermediate_future);
}

// this overload handles the special case where the user function returns void
template<class Executor, class Function, class Future, class Tuple, size_t... TupleIndices>
executor_future_t<Executor, void>
  bulk_then_executor_impl(Executor& exec,
                          Function f,
                          void_factory,
                          typename executor_traits<Executor>::shape_type shape,
                          Future& fut,
                          Tuple&& factory_tuple,
                          detail::index_sequence<TupleIndices...>)
{
  return executor_traits<Executor>::then_execute(exec, f, shape, fut, detail::get<TupleIndices>(std::forward<Tuple>(factory_tuple))...);
}


// XXX upon c++14, just return auto from these functions
template<class Result, class Future, class Function>
struct unpack_shared_parameters_from_then_execute_and_invoke
{
  mutable Function f;

  // this overload of impl() handles the case when the future given to then_execute() is non-void
  template<size_t... TupleIndices, class Index, class PastArg, class Tuple>
  __AGENCY_ANNOTATION
  Result impl(detail::index_sequence<TupleIndices...>, const Index& idx, PastArg& past_arg, Tuple&& tuple_of_shared_args) const
  {
    return f(idx, past_arg, detail::get<TupleIndices>(tuple_of_shared_args)...);
  }

  // this overload of impl() handles the case when the future given to then_execute() is void
  template<size_t... TupleIndices, class Index, class Tuple>
  __AGENCY_ANNOTATION
  Result impl(detail::index_sequence<TupleIndices...>, const Index& idx, Tuple&& tuple_of_shared_args) const
  {
    return f(idx, detail::get<TupleIndices>(tuple_of_shared_args)...);
  }

  // this overload of operator() handles the case when the future given to then_execute() is non-void
  template<class Index, class PastArg, class... Types,
           class Future1 = Future,
           class = typename std::enable_if<
             is_non_void_future<Future1>::value
           >::type>
  __AGENCY_ANNOTATION
  Result operator()(const Index& idx, PastArg& past_arg, Types&... packaged_shared_args) const
  {
    // unpack the packaged shared parameters into a tuple
    auto tuple_of_shared_args = detail::unpack_shared_parameters_from_executor(packaged_shared_args...);

    return impl(detail::make_tuple_indices(tuple_of_shared_args), idx, past_arg, tuple_of_shared_args);
  }


  // this overload of operator() handles the case when the future given to then_execute() is void
  template<class Index, class... Types,
           class Future1 = Future,
           class = typename std::enable_if<
             is_void_future<Future1>::value
           >::type>
  __AGENCY_ANNOTATION
  Result operator()(const Index& idx, Types&... packaged_shared_args) const
  {
    // unpack the packaged shared parameters into a tuple
    auto tuple_of_shared_args = detail::unpack_shared_parameters_from_executor(packaged_shared_args...);

    return impl(detail::make_tuple_indices(tuple_of_shared_args), idx, tuple_of_shared_args);
  }
};

template<class Result, class Future, class Function>
__AGENCY_ANNOTATION
unpack_shared_parameters_from_then_execute_and_invoke<Result, Future, Function> make_unpack_shared_parameters_from_then_execute_and_invoke(Function f)
{
  return unpack_shared_parameters_from_then_execute_and_invoke<Result, Future, Function>{f};
}


// computes the result type of bulk_then(executor)
template<class Executor, class Function, class Future, class... Args>
struct bulk_then_executor_result
{
  // figure out the Future's value_type
  using future_value_type = typename future_traits<Future>::value_type;

  // assemble a list of template parameters for bulk_async_executor_result
  // when Future is a void future, we don't want to include it in the list
  using template_parameters = typename std::conditional<
    is_void_future<Future>::value,
    type_list<Executor,Function,Args...>,
    type_list<Executor,Function,Future,Args...>
  >::type;

  // to compute the result of bulk_then_executor(), instantiate
  // bulk_async_executor_result_t with the list of template parameters
  using type = type_list_instantiate<bulk_async_executor_result_t, template_parameters>;
};

template<class Executor, class Function, class Future, class... Args>
using bulk_then_executor_result_t = typename bulk_then_executor_result<Executor,Function,Future,Args...>::type;


template<class Future,
         class Function,
         class... Args,
         class = typename std::enable_if<
           is_non_void_future<Future>::value
         >::type>
__AGENCY_ANNOTATION
auto bind_agent_local_parameters_for_bulk_then(Function f, Args&&... args) ->
  decltype(detail::bind_agent_local_parameters_workaround_nvbug1754712(std::integral_constant<size_t,2>(), f, detail::placeholders::_1, detail::placeholders::_2, std::forward<Args>(args)...))
{
  // the _1 is for the executor idx parameter, which is the first parameter passed to f
  // the _2 is for the future parameter, which is the second parameter passed to f
  // the agent local parameters begin at index 2
  return detail::bind_agent_local_parameters_workaround_nvbug1754712(std::integral_constant<size_t,2>(), f, detail::placeholders::_1, detail::placeholders::_2, std::forward<Args>(args)...);
}

template<class Future,
         class Function,
         class... Args,
         class = typename std::enable_if<
           is_void_future<Future>::value
         >::type>
__AGENCY_ANNOTATION
auto bind_agent_local_parameters_for_bulk_then(Function f, Args&&... args) ->
  decltype(detail::bind_agent_local_parameters_workaround_nvbug1754712(std::integral_constant<size_t,1>(), f, detail::placeholders::_1, std::forward<Args>(args)...))
{
  // the _1 is for the executor idx parameter, which is the first parameter passed to f
  // the Future is void, so we don't have to reserve a parameter slot for its (non-existent) value
  // the agent local parameters begin at index 1
  return detail::bind_agent_local_parameters_workaround_nvbug1754712(std::integral_constant<size_t,1>(), f, detail::placeholders::_1, std::forward<Args>(args)...);
}


template<class Executor, class Function, class Future, class... Args>
bulk_then_executor_result_t<Executor,Function,Future,Args...>
  bulk_then_executor(Executor& exec, typename executor_traits<typename std::decay<Executor>::type>::shape_type shape, Function f, Future& fut, Args&&... args)
{
  // bind f and the agent local parameters in args... into a functor g
  auto g = detail::bind_agent_local_parameters_for_bulk_then<Future>(f, std::forward<Args>(args)...);

  // make a tuple of the shared args
  auto shared_arg_tuple = detail::forward_shared_parameters_as_tuple(std::forward<Args>(args)...);

  using traits = executor_traits<Executor>;

  // package up the shared parameters for the executor
  const size_t execution_depth = traits::execution_depth;

  // create a tuple of factories to use for shared parameters for the executor
  auto factory_tuple = agency::detail::make_shared_parameter_factory_tuple<execution_depth>(shared_arg_tuple);

  // compute the type of f's result
  using result_of_f = detail::result_of_continuation_t<Function,executor_index_t<Executor>,Future,decay_parameter_t<Args>...>;

  // unpack shared parameters we receive from the executor
  auto h = detail::make_unpack_shared_parameters_from_then_execute_and_invoke<result_of_f,Future>(g);

  // based on the type of f's result, make a factory that will create the appropriate type of container to store f's results
  auto result_factory = detail::make_result_factory<result_of_f>(exec);

  return detail::bulk_then_executor_impl(exec, h, result_factory, shape, fut, factory_tuple, detail::make_index_sequence<execution_depth>());
}


template<class ExecutorTraits, class AgentTraits, class Function, class Future, size_t... UserArgIndices>
struct then_execute_agent_functor
{
  // XXX should just make the future's value_type a parameter of this functor and try to use it SFINAE the operator()s below
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
  __AGENCY_ANNOTATION
  static result_of_t<OtherFunction(agent_type&)>
    unpack_shared_params_and_execute(OtherFunction f, const agent_index_type& index, const agent_param_type& param, Tuple&& shared_params, detail::index_sequence<Indices...>)
  {
    return AgentTraits::execute(f, index, param, detail::get<Indices>(std::forward<Tuple>(shared_params))...);
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
    auto args_tuple = detail::forward_as_tuple(std::forward<Args>(args)...);

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
      return f_(self, past_arg, detail::get<UserArgIndices>(user_args)...);
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

    return this->unpack_shared_params_and_execute(invoke_f, agent_idx, agent_param_, agent_shared_args, detail::make_tuple_indices(agent_shared_args));
  }
};


template<class ExecutionPolicy, class Function, class Future, class... Args>
struct bulk_then_execution_policy_result
{
  // figure out the Future's value_type
  using future_value_type = typename future_traits<Future>::value_type;

  // avoid passing Future to bulk_invoke_execution_policy_result when it is a void Future
  using bulk_invoke_result_type = typename detail::lazy_conditional<
    std::is_void<future_value_type>::value,
    bulk_invoke_execution_policy_result<ExecutionPolicy,Function,Args...>,
    bulk_invoke_execution_policy_result<ExecutionPolicy,Function,Future,Args...>
  >::type;

  using type = policy_future_t<
    ExecutionPolicy,
    bulk_invoke_result_type
  >;
};

template<class ExecutionPolicy, class Function, class Future, class... Args>
using bulk_then_execution_policy_result_t = typename bulk_then_execution_policy_result<ExecutionPolicy,Function,Future,Args...>::type;


template<size_t... UserArgIndices, size_t... SharedArgIndices, class ExecutionPolicy, class Function, class Future, class... Args>
bulk_then_execution_policy_result_t<
  ExecutionPolicy, Function, Future, Args...
>
  bulk_then_execution_policy_impl(index_sequence<UserArgIndices...>,
                                  index_sequence<SharedArgIndices...>,
                                  ExecutionPolicy& policy, Function f, Future& fut, Args&&... args)
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
  auto lambda = then_execute_agent_functor<executor_traits,agent_traits,Function,Future,UserArgIndices...>{param, agent_shape, executor_shape, f};

  return detail::bulk_then_executor(policy.executor(), executor_shape, lambda, fut, std::forward<Args>(args)..., agency::share_at_scope<SharedArgIndices>(detail::get<SharedArgIndices>(agent_shared_param_tuple))...);
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
  static_assert(!detail::is_cuda_extended_device_lambda<Function>::value, "CUDA extended device lambdas are not supported by bulk_then().");

  using agent_traits = execution_agent_traits<typename std::decay<ExecutionPolicy>::type::execution_agent_type>;
  const size_t num_shared_params_for_agent = detail::execution_depth<typename agent_traits::execution_category>::value;

  return detail::bulk_then_execution_policy_impl(
    detail::index_sequence_for<Args...>(),
    detail::make_index_sequence<num_shared_params_for_agent>(),
    policy,
    f,
    fut,
    std::forward<Args>(args)...
  );
}


} // end agency

