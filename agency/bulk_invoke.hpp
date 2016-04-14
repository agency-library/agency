#pragma once

#include <agency/detail/config.hpp>
#include <agency/executor_traits.hpp>
#include <agency/execution_agent.hpp>
#include <agency/functional.hpp>
#include <agency/detail/bulk_invoke/shared_parameter.hpp>
#include <agency/detail/bulk_invoke/bind_agent_local_parameters.hpp>
#include <agency/detail/is_call_possible.hpp>
#include <agency/detail/shape_cast.hpp>
#include <agency/detail/index_cast.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/execution_policy_traits.hpp>
#include <agency/detail/bulk_invoke/single_result.hpp>
#include <agency/detail/bulk_invoke/result_factory.hpp>


// XXX this has gotten complicated, so we should reorganize the implementation of bulk_invoke()
//     into headers organized under agency/detail/bulk_invoke/*.hpp
namespace agency
{


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


// this overload handles the general case where the user function returns a normal result
template<class Executor, class Function, class Factory, class Tuple, size_t... TupleIndices>
typename std::result_of<Factory(executor_shape_t<Executor>)>::type
  bulk_invoke_executor_impl(Executor& exec,
                            Function f,
                            Factory result_factory,
                            typename executor_traits<Executor>::shape_type shape,
                            Tuple&& factory_tuple,
                            detail::index_sequence<TupleIndices...>)
{
  return executor_traits<Executor>::execute(exec, f, result_factory, shape, detail::get<TupleIndices>(std::forward<Tuple>(factory_tuple))...);
}

// this overload handles the special case where the user function returns a scope_result
template<class Executor, class Function, size_t scope, class T, class Tuple, size_t... TupleIndices>
typename detail::scope_result_container<scope,T,Executor>::result_type
  bulk_invoke_executor_impl(Executor& exec,
                            Function f,
                            detail::executor_traits_detail::container_factory<detail::scope_result_container<scope,T,Executor>> result_factory,
                            typename executor_traits<Executor>::shape_type shape,
                            Tuple&& factory_tuple,
                            detail::index_sequence<TupleIndices...>)
{
  return executor_traits<Executor>::execute(exec, f, result_factory, shape, detail::get<TupleIndices>(std::forward<Tuple>(factory_tuple))...);
}

// this overload handles the special case where the user function returns void
template<class Executor, class Function, class Tuple, size_t... TupleIndices>
void bulk_invoke_executor_impl(Executor& exec,
                               Function f,
                               void_factory,
                               typename executor_traits<Executor>::shape_type shape,
                               Tuple&& factory_tuple,
                               detail::index_sequence<TupleIndices...>)
{
  return executor_traits<Executor>::execute(exec, f, shape, detail::get<TupleIndices>(std::forward<Tuple>(factory_tuple))...);
}


// this metafunction computes the type of the parameter passed to a user function
// given then type of parameter passed to bulk_invoke/bulk_async/etc.
template<class T>
struct decay_parameter
{
  template<class U>
  struct lazy_add_lvalue_reference
  {
    using type = typename std::add_lvalue_reference<typename U::type>::type;
  };

  // first decay the parameter
  using decayed_type = typename std::decay<T>::type;

  // when passing a parameter to the user's function:
  // if the parameter is a future, then we pass a reference to its value type
  // otherwise, we pass a copy of the decayed_type
  using type = typename detail::lazy_conditional<
    is_future<decayed_type>::value,
    lazy_add_lvalue_reference<future_value<decayed_type>>,
    identity<decayed_type>
  >::type;
};

template<class T>
using decay_parameter_t = typename decay_parameter<T>::type;


template<size_t level, class T, class... Args>
struct decay_parameter<shared_parameter<level,T,Args...>>
{
  // shared_parameters are passed to the user function by reference
  using type = T&;
};


// computes the result type of bulk_invoke(executor)
template<class Executor, class Function, class... Args>
struct bulk_invoke_executor_result
{
  // first figure out what type the user function returns
  using user_function_result = result_of_t<
    Function(executor_index_t<Executor>, decay_parameter_t<Args>...)
  >;

  // if the user function returns scope_result, then use scope_result_to_bulk_invoke_result to figure out what to return
  // else, the result is whatever executor_result<Executor, function_result> thinks it is
  using type = typename detail::lazy_conditional<
    detail::is_scope_result<user_function_result>::value,
    detail::scope_result_to_bulk_invoke_result<user_function_result, Executor>,
    executor_result<Executor, user_function_result>
  >::type;
};

template<class Executor, class Function, class... Args>
using bulk_invoke_executor_result_t = typename bulk_invoke_executor_result<Executor,Function,Args...>::type;


template<class Executor, class Function, class... Args>
bulk_invoke_executor_result_t<Executor, Function, Args...>
  bulk_invoke_executor(Executor& exec, typename executor_traits<Executor>::shape_type shape, Function f, Args&&... args)
{
  // the _1 is for the executor idx parameter, which is the first parameter passed to f
  auto g = detail::bind_agent_local_parameters_workaround_nvbug1754712(std::integral_constant<size_t,1>(), f, detail::placeholders::_1, std::forward<Args>(args)...);

  // make a tuple of the shared args
  auto shared_arg_tuple = detail::forward_shared_parameters_as_tuple(std::forward<Args>(args)...);

  using traits = executor_traits<Executor>;

  // package up the shared parameters for the executor
  const size_t execution_depth = traits::execution_depth;

  // create a tuple of factories to use for shared parameters for the executor
  auto factory_tuple = detail::make_shared_parameter_factory_tuple<execution_depth>(shared_arg_tuple);

  // unpack shared parameters we receive from the executor
  auto h = detail::make_unpack_shared_parameters_from_executor_and_invoke(g);

  // compute the type of f's result
  using result_of_f = typename std::result_of<Function(executor_index_t<Executor>,decay_parameter_t<Args>...)>::type;

  // based on the type of f's result, make a factory that will create the appropriate type of container to store f's results
  auto result_factory = detail::make_result_factory<result_of_f>(exec);

  return detail::bulk_invoke_executor_impl(exec, h, result_factory, shape, factory_tuple, detail::make_index_sequence<execution_depth>());
}


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
  __AGENCY_ANNOTATION
  static result_of_t<OtherFunction(agent_type&)>
    unpack_shared_params_and_execute(OtherFunction f, const agent_index_type& index, const agent_param_type& param, Tuple&& shared_params, detail::index_sequence<Indices...>)
  {
    return AgentTraits::execute(f, index, param, detail::get<Indices>(std::forward<Tuple>(shared_params))...);
  }

  // execution_agent_traits::execute expects a function whose only parameter is agent_type
  // so to invoke the user function with it, we have to create a function of one parameter by
  // binding the user's function and arguments together into this functor
  template<class Tuple, size_t... Indices>
  struct unpack_arguments_and_invoke_with_self
  {
    Function& f_;
    Tuple& args_;

    template<class ExecutionAgent>
    __AGENCY_ANNOTATION
    auto operator()(ExecutionAgent& self) ->
      decltype(
        f_(self, agency::detail::get<Indices>(args_)...)
      )
    {
      return f_(self, agency::detail::get<Indices>(args_)...);
    }
  };

  template<class... Args>
  __AGENCY_ANNOTATION
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
    //auto invoke_f = [&user_args,this] (agent_type& self)
    //{
    //  // invoke f by passing the agent, then the user's parameters
    //  return f_(self, detail::get<UserArgIndices>(user_args)...);
    //};
    // XXX seems like we could do this with a bind() instead of introducing a named functor
    auto invoke_f = unpack_arguments_and_invoke_with_self<decltype(user_args), UserArgIndices...>{f_,user_args};

    constexpr size_t num_shared_args = std::tuple_size<decltype(agent_shared_args)>::value;
    return this->unpack_shared_params_and_execute(invoke_f, agent_idx, agent_param_, agent_shared_args, detail::make_index_sequence<num_shared_args>());
  }
};


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

  return detail::bulk_invoke_executor(policy.executor(), executor_shape, lambda, std::forward<Args>(args)..., agency::share_at_scope<SharedArgIndices>(detail::get<SharedArgIndices>(agent_shared_param_tuple))...);
}


template<class ExecutionPolicy, class Function, class... Args>
struct is_bulk_call_possible_via_execution_policy_impl
{
  template<class ExecutionPolicy1, class Function1, class... Args1,
           class = typename std::enable_if<
             has_execution_agent_type<ExecutionPolicy1>::value
           >::type,
           class = typename enable_if_call_possible<
             void, Function1, execution_policy_agent_t<ExecutionPolicy1>&, decay_parameter_t<Args1>...
           >::type
          >
  static std::true_type test(int);

  template<class...>
  static std::false_type test(...);

  using type = decltype(test<ExecutionPolicy,Function,Args...>(0));
};

template<class ExecutionPolicy, class Function, class... Args>
using is_bulk_call_possible_via_execution_policy = typename is_bulk_call_possible_via_execution_policy_impl<ExecutionPolicy,Function,Args...>::type;


template<bool enable, class ExecutionPolicy, class Function, class... Args>
struct enable_if_bulk_invoke_execution_policy_impl {};

template<class ExecutionPolicy, class Function, class... Args>
struct enable_if_bulk_invoke_execution_policy_impl<true, ExecutionPolicy, Function, Args...>
{
  using type = bulk_invoke_execution_policy_result_t<ExecutionPolicy,Function,Args...>;
};

template<class ExecutionPolicy, class Function, class... Args>
struct enable_if_bulk_invoke_execution_policy
  : enable_if_bulk_invoke_execution_policy_impl<
      is_bulk_call_possible_via_execution_policy<decay_t<ExecutionPolicy>,Function,Args...>::value,
      decay_t<ExecutionPolicy>,
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


} // end agency

#include <agency/detail/bulk_invoke/bulk_async.hpp>

// XXX uncomment this when bulk_then is ready
//#include <agency/detail/bulk_invoke/bulk_then.hpp>

