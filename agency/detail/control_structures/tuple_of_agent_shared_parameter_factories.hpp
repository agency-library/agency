#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/execution_agent/execution_agent_traits.hpp>
#include <agency/execution/executor/properties/detail/bulk_guarantee_depth.hpp>
#include <agency/tuple.hpp>
#include <agency/detail/type_list.hpp>
#include <agency/detail/control_structures/agent_shared_parameter_factory.hpp>
#include <agency/detail/scoped_in_place_type.hpp>


namespace agency
{
namespace detail
{


template<class AgentList, class... Barriers>
struct tuple_of_agent_shared_parameter_factories_impl;

template<class... ExecutionAgents, class... Barriers>
struct tuple_of_agent_shared_parameter_factories_impl<agency::detail::type_list<ExecutionAgents...>, Barriers...>
{
  using type = agency::tuple<
    agent_shared_parameter_factory<ExecutionAgents,Barriers>...
  >;
};

template<class ExecutionAgent, class... Barriers>
struct tuple_of_agent_shared_parameter_factories : tuple_of_agent_shared_parameter_factories_impl<
  typename agency::detail::execution_agent_type_list<ExecutionAgent>::type,
  Barriers...
>
{};

template<class ExecutionAgent, class... Barriers>
using tuple_of_agent_shared_parameter_factories_t = typename tuple_of_agent_shared_parameter_factories<ExecutionAgent, Barriers...>::type;


// this is the terminal case for flat agents -- agents which have no ::inner_execution_agent_type
template<class ExecutionAgent,
         class BarrierOrVoid,
         __AGENCY_REQUIRES(
           !agency::detail::has_inner_execution_agent_type<ExecutionAgent>::value
        )>
__AGENCY_ANNOTATION
tuple_of_agent_shared_parameter_factories_t<ExecutionAgent, BarrierOrVoid>
  make_tuple_of_agent_shared_parameter_factories(const typename agency::execution_agent_traits<ExecutionAgent>::param_type& param,
                                                 scoped_in_place_type_t<BarrierOrVoid> barrier)
{
  auto factory = make_agent_shared_parameter_factory<ExecutionAgent>(param, barrier.outer());

  return agency::make_tuple(factory);
}

// this is the recursive case for scoped agents -- agents which do have an ::inner_execution_agent_type
template<class ExecutionAgent,
         class BarrierOrVoid, class... BarriersOrVoids,
         __AGENCY_REQUIRES(
           1 + sizeof...(BarriersOrVoids) == agency::detail::bulk_guarantee_depth<typename agency::execution_agent_traits<ExecutionAgent>::execution_requirement>::value
         ),
         __AGENCY_REQUIRES(
           agency::detail::has_inner_execution_agent_type<ExecutionAgent>::value
         )>
__AGENCY_ANNOTATION
tuple_of_agent_shared_parameter_factories_t<ExecutionAgent, BarrierOrVoid, BarriersOrVoids...>
  make_tuple_of_agent_shared_parameter_factories(const typename agency::execution_agent_traits<ExecutionAgent>::param_type& param,
                                                 scoped_in_place_type_t<BarrierOrVoid,BarriersOrVoids...> barriers)
{
  using inner_execution_agent_type = typename ExecutionAgent::inner_execution_agent_type;

  // recurse to get the tail of the tuple
  auto inner_factories = make_tuple_of_agent_shared_parameter_factories<inner_execution_agent_type>(param.inner(), barriers.inner());

  // create the head of the tuple
  auto outer_factory = make_agent_shared_parameter_factory<ExecutionAgent>(param, barriers.outer());

  // return a tuple of all the factories by prepending the outer_factory
  return agency::tuple_cat(agency::make_tuple(outer_factory), inner_factories);
}


} // end detail
} // end agency

