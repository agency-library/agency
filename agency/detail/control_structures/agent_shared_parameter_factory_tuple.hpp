#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/factory.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/execution/execution_agent.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{


// when ExecutionAgent has a shared_param_type, we make a factory that returns ::shared_param_type
template<class ExecutionAgent,
         class = typename std::enable_if<
           detail::has_shared_param_type<ExecutionAgent>::value
         >::type>
__AGENCY_ANNOTATION
detail::construct<typename ExecutionAgent::shared_param_type, typename execution_agent_traits<ExecutionAgent>::param_type>
  make_agent_shared_parameter_factory(const typename execution_agent_traits<ExecutionAgent>::param_type& param)
{
  return detail::make_construct<typename ExecutionAgent::shared_param_type>(param);
}

// when ExecutionAgent does not have a shared_param_type, we make a factory that returns ignore_t
template<class ExecutionAgent,
         class = typename std::enable_if<
           !detail::has_shared_param_type<ExecutionAgent>::value
         >::type>
__AGENCY_ANNOTATION
detail::construct<agency::detail::ignore_t>
  make_agent_shared_parameter_factory(const typename execution_agent_traits<ExecutionAgent>::param_type&)
{
  return detail::make_construct<agency::detail::ignore_t>();
}


// give the return type of make_agent_shared_parameter_factory<ExecutionAgent>() a name
template<class ExecutionAgent>
using agent_shared_parameter_factory = decltype(make_agent_shared_parameter_factory<ExecutionAgent>(std::declval<typename execution_agent_traits<ExecutionAgent>::param_type>()));



template<class TypeList>
struct agent_shared_parameter_factory_tuple_impl;

template<class... ExecutionAgents>
struct agent_shared_parameter_factory_tuple_impl<detail::type_list<ExecutionAgents...>>
{
  using type = detail::tuple<
    agent_shared_parameter_factory<ExecutionAgents>...
  >;
};

template<class ExecutionAgent>
struct agent_shared_parameter_factory_tuple : agent_shared_parameter_factory_tuple_impl<
  typename detail::execution_agent_type_list<ExecutionAgent>::type
>
{};

template<class ExecutionAgent>
using agent_shared_parameter_factory_tuple_t = typename agent_shared_parameter_factory_tuple<ExecutionAgent>::type;


// this function creates a tuple where each tuple element is a factory
// each element of the tuple corresponds to a level in ExecutionAgent's execution hierarchy
// each factory creates an execution level's shared_param_type using its corresponding param_type
template<class ExecutionAgent>
__AGENCY_ANNOTATION
agent_shared_parameter_factory_tuple_t<ExecutionAgent>
  make_agent_shared_parameter_factory_tuple(const typename execution_agent_traits<ExecutionAgent>::param_type& param);


// this is the terminal case for flat agents -- agents which have no ::inner_execution_agent_type
template<class ExecutionAgent>
__AGENCY_ANNOTATION
agent_shared_parameter_factory_tuple_t<ExecutionAgent>
  make_agent_shared_parameter_factory_tuple_impl(const typename execution_agent_traits<ExecutionAgent>::param_type& param, std::false_type)
{
  using shared_param_type = typename execution_agent_traits<ExecutionAgent>::shared_param_type;

  auto factory = detail::make_agent_shared_parameter_factory<ExecutionAgent>(param);

  return detail::make_tuple(factory);
}

// this is the recursive case for scoped agents -- agents which do have an ::inner_execution_agent_type
template<class ExecutionAgent>
__AGENCY_ANNOTATION
agent_shared_parameter_factory_tuple_t<ExecutionAgent>
  make_agent_shared_parameter_factory_tuple_impl(const typename execution_agent_traits<ExecutionAgent>::param_type& param, std::true_type)
{
  using inner_execution_agent_type = typename ExecutionAgent::inner_execution_agent_type;

  // recurse to get the tail of the tuple
  auto inner_factories = make_agent_shared_parameter_factory_tuple<inner_execution_agent_type>(param.inner());

  // create the head of the tuple
  using shared_param_type = typename execution_agent_traits<ExecutionAgent>::shared_param_type;

  auto outer_factory = detail::make_agent_shared_parameter_factory<ExecutionAgent>(param);

  // prepend the head
  return __tu::tuple_prepend_invoke(inner_factories, outer_factory, detail::agency_tuple_maker());
}


template<class ExecutionAgent>
__AGENCY_ANNOTATION
agent_shared_parameter_factory_tuple_t<ExecutionAgent>
  make_agent_shared_parameter_factory_tuple(const typename execution_agent_traits<ExecutionAgent>::param_type& param)
{
  return detail::make_agent_shared_parameter_factory_tuple_impl<ExecutionAgent>(param, typename detail::has_inner_execution_agent_type<ExecutionAgent>::type());
}


} // end detail
} // end agency

