#pragma once

#include <agency/detail/config.hpp>
#include <agency/tuple.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/coordinate/detail/index_cast.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/execution/executor/executor_traits/executor_shape.hpp>
#include <agency/execution/executor/executor_traits/executor_index.hpp>
#include <utility>

namespace agency
{
namespace detail
{


template<class Executor, class AgentTraits, class Function, size_t... UserArgIndices>
struct execute_agent_functor
{
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

  // execution_agent_traits::execute expects a function whose only parameter is agent_type
  // so to invoke the user function with it, we have to create a function of one parameter by
  // binding the user's function and arguments together into this functor
  template<class Tuple, size_t... Indices>
  struct unpack_arguments_and_invoke_with_self
  {
    Function& f_;
    Tuple& args_;

    __AGENCY_ANNOTATION
    result_of_t<Function(agent_type&,typename std::tuple_element<Indices,Tuple>::type...)>
      operator()(agent_type& self)
    {
      using result_type = result_of_t<Function(agent_type&,typename std::tuple_element<Indices,Tuple>::type...)>;

      // XXX we explicitly cast the result of f_ to result_type
      //     this is due to our special implementation of result_of,
      //     coerces the return type of a CUDA extended device lambda to be void
      return static_cast<result_type>(agency::detail::invoke(f_, self, agency::get<Indices>(args_)...));
    }
  };

  template<class... Args>
  __AGENCY_ANNOTATION
  result_of_t<Function(agent_type&, pack_element_t<UserArgIndices, Args&&...>...)>
    operator()(const executor_index_type& executor_idx, Args&&... args)
  {
    // collect all parameters into a tuple of references
    auto args_tuple = agency::forward_as_tuple(std::forward<Args>(args)...);

    // split the parameters into user parameters and agent parameters
    auto user_args         = detail::tuple_take_view<sizeof...(UserArgIndices)>(args_tuple);
    auto agent_shared_args = detail::tuple_drop_view<sizeof...(UserArgIndices)>(args_tuple);

    // turn the executor index into an agent index
    // XXX ideally, bulk_invoke et al should require() an executor which produces the right type of indices over the right domain
    auto agent_idx = AgentTraits::domain(agent_param_).origin() + detail::index_cast<agent_index_type>(executor_idx, executor_shape_, agent_shape_);

    // AgentTraits::execute expects a function whose only parameter is agent_type
    // so we have to wrap f_ into a function of one parameter
    //auto invoke_f = [&user_args,this] (agent_type& self)
    //{
    //  // invoke f by passing the agent, then the user's parameters
    //  return f_(self, agency::get<UserArgIndices>(user_args)...);
    //};
    // XXX seems like we could do this with a bind() instead of introducing a named functor
    auto invoke_f = unpack_arguments_and_invoke_with_self<decltype(user_args), UserArgIndices...>{f_,user_args};

    constexpr size_t num_shared_args = std::tuple_size<decltype(agent_shared_args)>::value;
    return this->unpack_shared_params_and_execute(invoke_f, agent_idx, agent_param_, agent_shared_args, detail::make_index_sequence<num_shared_args>());
  }
};




} // end detail
} // end agency

