#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/bulk_invoke/bind_agent_local_parameters.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/executor_traits.hpp>
#include <agency/execution_agent.hpp>
#include <agency/bulk_invoke.hpp>
#include <utility>
#include <tuple>
#include <type_traits>


namespace agency
{
namespace cuda
{
namespace detail
{

  
template<class Function>
struct unpack_shared_parameters_from_executor_and_invoke
{
  mutable Function g;

  template<class Index, class Tuple, size_t... I>
  __AGENCY_ANNOTATION
  auto invoke_from_tuple_impl(const Index& idx, const Tuple& t, agency::detail::index_sequence<I...>) const
    -> decltype(
         g(idx, agency::detail::get<I>(t)...)
       )
  {
    return g(idx, agency::detail::get<I>(t)...);
  }

  template<class Index, class Tuple>
  __AGENCY_ANNOTATION
  auto invoke_from_tuple(const Index& idx, const Tuple& t) const
    -> decltype(
         this->invoke_from_tuple_impl(idx, t, agency::detail::make_index_sequence<std::tuple_size<Tuple>::value>())
       )
  {
    return this->invoke_from_tuple_impl(idx, t, agency::detail::make_index_sequence<std::tuple_size<Tuple>::value>());
  }

  template<class Index, class... Types>
  __AGENCY_ANNOTATION
  auto operator()(const Index& idx, Types&... packaged_shared_params) const
    -> decltype(
         this->invoke_from_tuple(
           idx,
           agency::detail::unpack_shared_parameters_from_executor(packaged_shared_params...)
         )
       )
  {
    auto shared_param_tuple = agency::detail::unpack_shared_parameters_from_executor(packaged_shared_params...);

    return this->invoke_from_tuple(idx, shared_param_tuple);
  }
};


} // end detail


template<class Executor, class Function, class... Args>
typename agency::detail::enable_if_bulk_invoke_executor<
  Executor, Function, Args...
>::type
  bulk_invoke(Executor& exec, typename executor_traits<typename std::decay<Executor>::type>::shape_type shape, Function f, Args&&... args)
{
  // the _1 is for the executor idx parameter, which is the first parameter passed to f
  auto g = agency::detail::bind_agent_local_parameters_workaround_nvbug1754712(std::integral_constant<size_t,1>(), f, agency::detail::placeholders::_1, std::forward<Args>(args)...);

  // make a tuple of the shared args
  auto shared_arg_tuple = agency::detail::forward_shared_parameters_as_tuple(std::forward<Args>(args)...);

  using traits = executor_traits<Executor>;

  // package up the shared parameters for the executor
  const size_t execution_depth = traits::execution_depth;

  // create a tuple of factories to use for shared parameters for the executor
  auto factory_tuple = agency::detail::make_shared_parameter_factory_tuple<execution_depth>(shared_arg_tuple);

  // unpack shared parameters we receive from the executor
  auto h = detail::unpack_shared_parameters_from_executor_and_invoke<decltype(g)>{g};

  // compute the type of f's result
  using result_of_f = typename std::result_of<Function(agency::detail::executor_index_t<Executor>,agency::detail::decay_parameter_t<Args>...)>::type;

  // based on the type of f's result, make a factory that will create the appropriate type of container to store f's results
  auto result_factory = agency::detail::make_result_factory<result_of_f>(exec);

  return agency::detail::bulk_invoke_executor_impl(exec, h, result_factory, shape, factory_tuple, agency::detail::make_index_sequence<execution_depth>());
}


template<class Executor, class Function, class... Args>
typename agency::detail::enable_if_bulk_async_executor<
  Executor, Function, Args...
>::type
  bulk_async(Executor& exec, typename executor_traits<typename std::decay<Executor>::type>::shape_type shape, Function f, Args&&... args)
{
  // the _1 is for the executor idx parameter, which is the first parameter passed to f
  auto g = agency::detail::bind_agent_local_parameters_workaround_nvbug1754712(std::integral_constant<size_t,1>(), f, agency::detail::placeholders::_1, std::forward<Args>(args)...);

  // make a tuple of the shared args
  auto shared_arg_tuple = agency::detail::forward_shared_parameters_as_tuple(std::forward<Args>(args)...);

  using traits = executor_traits<Executor>;

  // package up the shared parameters for the executor
  const size_t execution_depth = traits::execution_depth;

  // create a tuple of factories to use for shared parameters for the executor
  auto factory_tuple = agency::detail::make_shared_parameter_factory_tuple<execution_depth>(shared_arg_tuple);

  // unpack shared parameters we receive from the executor
  auto h = detail::unpack_shared_parameters_from_executor_and_invoke<decltype(g)>{g};

  // compute the type of f's result
  using result_of_f = typename std::result_of<Function(agency::detail::executor_index_t<Executor>,agency::detail::decay_parameter_t<Args>...)>::type;

  // based on the type of f's result, make a factory that will create the appropriate type of container to store f's results
  auto result_factory = agency::detail::make_result_factory<result_of_f>(exec);

  return agency::detail::bulk_async_executor_impl(exec, h, result_factory, shape, factory_tuple, agency::detail::make_index_sequence<execution_depth>());
}


namespace detail
{


template<class Function, class Tuple, size_t... Indices>
struct unpack_arguments_and_invoke_with_self
{
  Function& f;
  Tuple& args;

  template<class ExecutionAgent>
  __AGENCY_ANNOTATION
  auto operator()(ExecutionAgent& self)
    -> decltype(
         f(self, agency::detail::get<Indices>(args)...)
       )
  {
    return f(self, agency::detail::get<Indices>(args)...);
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

  template<class OtherFunction, class Tuple, size_t... Indices>
  __AGENCY_ANNOTATION
  static agency::detail::result_of_t<OtherFunction(agent_type&)>
    unpack_shared_params_and_execute(OtherFunction f, const agent_index_type& index, const agent_param_type& param, Tuple&& shared_params, agency::detail::index_sequence<Indices...>)
  {
    return AgentTraits::execute(f, index, param, detail::get<Indices>(std::forward<Tuple>(shared_params))...);
  }

  template<class... Args>
  __AGENCY_ANNOTATION
  agency::detail::result_of_t<Function(agent_type&, agency::detail::pack_element_t<UserArgIndices, Args&&...>...)>
    operator()(const executor_index_type& executor_idx, Args&&... args)
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

    constexpr size_t num_shared_args = std::tuple_size<decltype(agent_shared_args)>::value;
    return this->unpack_shared_params_and_execute(invoke_f, agent_idx, agent_param_, agent_shared_args, agency::detail::make_index_sequence<num_shared_args>());
  }
};


template<size_t... UserArgIndices, size_t... SharedArgIndices, class ExecutionPolicy, class Function, class... Args>
agency::detail::bulk_invoke_execution_policy_result_t<
  ExecutionPolicy, Function, Args...
>
  bulk_invoke_execution_policy(agency::detail::index_sequence<UserArgIndices...>,
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

  return agency::cuda::bulk_invoke(policy.executor(), executor_shape, invoke_me, std::forward<Args>(args)..., agency::share_at_scope<SharedArgIndices>(agency::detail::get<SharedArgIndices>(agent_shared_param_tuple))...);
}


template<size_t... UserArgIndices, size_t... SharedArgIndices, class ExecutionPolicy, class Function, class... Args>
agency::detail::bulk_async_execution_policy_result_t<
  ExecutionPolicy, Function, Args...
>
  bulk_async_execution_policy(agency::detail::index_sequence<UserArgIndices...>,
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

  return agency::cuda::bulk_async(policy.executor(), executor_shape, invoke_me, std::forward<Args>(args)..., agency::share_at_scope<SharedArgIndices>(agency::detail::get<SharedArgIndices>(agent_shared_param_tuple))...);
}


} // end detail


template<class ExecutionPolicy, class Function, class... Args>
typename agency::detail::enable_if_bulk_invoke_execution_policy<
  ExecutionPolicy, Function, Args...
>::type
  bulk_invoke(ExecutionPolicy&& policy, Function f, Args&&... args)
{
  using agent_traits = execution_agent_traits<typename std::decay<ExecutionPolicy>::type::execution_agent_type>;
  const size_t num_shared_params = agency::detail::execution_depth<typename agent_traits::execution_category>::value;

  return detail::bulk_invoke_execution_policy(agency::detail::index_sequence_for<Args...>(), agency::detail::make_index_sequence<num_shared_params>(), policy, f, std::forward<Args>(args)...);
}


template<class ExecutionPolicy, class Function, class... Args>
typename agency::detail::enable_if_bulk_async_execution_policy<
  ExecutionPolicy, Function, Args...
>::type
  bulk_async(ExecutionPolicy&& policy, Function&& f, Args&&... args)
{
  using agent_traits = execution_agent_traits<typename std::decay<ExecutionPolicy>::type::execution_agent_type>;
  const size_t num_shared_params = agency::detail::execution_depth<typename agent_traits::execution_category>::value;

  return detail::bulk_async_execution_policy(agency::detail::index_sequence_for<Args...>(), agency::detail::make_index_sequence<num_shared_params>(), policy, f, std::forward<Args>(args)...);
}


} // end cuda
} // end agency

