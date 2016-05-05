#pragma once

#include <agency/detail/config.hpp>
#include <agency/executor_traits.hpp>
#include <agency/execution_agent.hpp>
#include <agency/functional.hpp>
#include <agency/detail/bulk_functions/shared_parameter.hpp>
#include <agency/detail/bulk_functions/bind_agent_local_parameters.hpp>
#include <agency/detail/is_call_possible.hpp>
#include <agency/detail/shape_cast.hpp>
#include <agency/detail/index_cast.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/execution_policy_traits.hpp>
#include <agency/detail/bulk_functions/single_result.hpp>
#include <agency/detail/bulk_functions/result_factory.hpp>
#include <agency/detail/type_traits.hpp>


// XXX this has gotten complicated, so we should reorganize the implementation of bulk_invoke()
//     into headers organized under agency/detail/bulk_invoke/*.hpp
namespace agency
{
namespace detail
{


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
    result_of_t<Function(ExecutionAgent&,typename std::tuple_element<Indices,Tuple>::type...)>
      operator()(ExecutionAgent& self)
    {
      using result_type = result_of_t<Function(ExecutionAgent&,typename std::tuple_element<Indices,Tuple>::type...)>;

      // XXX we explicitly cast the result of f_ to result_type
      //     this is due to our special implementation of result_of,
      //     coerces the return type of a CUDA extended device lambda to be void
      return static_cast<result_type>(f_(self, agency::detail::get<Indices>(args_)...));
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


} // end detail
} // end agency

// XXX move these to the top of this header
#include <agency/detail/bulk_functions/bulk_invoke.hpp>
#include <agency/detail/bulk_functions/bulk_async.hpp>
#include <agency/detail/bulk_functions/bulk_then.hpp>

