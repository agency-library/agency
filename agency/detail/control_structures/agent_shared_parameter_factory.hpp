#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/experimental/variant.hpp>
#include <agency/execution/execution_agent/execution_agent_traits.hpp>
#include <agency/detail/factory.hpp>


namespace agency
{
namespace detail
{


// when ExecutionAgent has a shared_param_type, we make a factory that returns ::shared_param_type
// this is the case for when a non-void type was given as the barrier type
// in that case, *do* pass a barrier parameter to the shared_param_type constructor
template<class ExecutionAgent,
         class Barrier,
         __AGENCY_REQUIRES(
           !std::is_void<Barrier>::value and
           agency::detail::has_shared_param_type<ExecutionAgent>::value
         )>
__AGENCY_ANNOTATION
agency::detail::construct<
  typename ExecutionAgent::shared_param_type,
  typename agency::execution_agent_traits<ExecutionAgent>::param_type,
  agency::experimental::in_place_type_t<Barrier>
>
  make_agent_shared_parameter_factory(const typename agency::execution_agent_traits<ExecutionAgent>::param_type& param,
                                      agency::experimental::in_place_type_t<Barrier> barrier)
{
  return agency::detail::make_construct<typename ExecutionAgent::shared_param_type>(param, barrier);
}


// when ExecutionAgent has a shared_param_type, we make a factory that returns ::shared_param_type
// this is the case for when void was given as the barrier type
// in that case, *do not* pass a barrier parameter to the shared_param_type constructor
template<class ExecutionAgent,
         __AGENCY_REQUIRES(
           agency::detail::has_shared_param_type<ExecutionAgent>::value
         )>
__AGENCY_ANNOTATION
agency::detail::construct<
  typename ExecutionAgent::shared_param_type,
  typename agency::execution_agent_traits<ExecutionAgent>::param_type
>
  make_agent_shared_parameter_factory(const typename agency::execution_agent_traits<ExecutionAgent>::param_type& param,
                                      agency::experimental::in_place_type_t<void>)
{
  return agency::detail::make_construct<typename ExecutionAgent::shared_param_type>(param);
}


// when ExecutionAgent does not have a shared_param_type, we make a factory that returns ignore_t
// any additional arguments to this function are ignored
template<class ExecutionAgent,
         class BarrierOrVoid,
         __AGENCY_REQUIRES(
           !agency::detail::has_shared_param_type<ExecutionAgent>::value
         )>
__AGENCY_ANNOTATION
agency::detail::construct<agency::detail::ignore_t>
  make_agent_shared_parameter_factory(const typename agency::execution_agent_traits<ExecutionAgent>::param_type&,
                                      agency::experimental::in_place_type_t<BarrierOrVoid>)
{
  return agency::detail::make_construct<agency::detail::ignore_t>();
}


// give the return type of make_agent_shared_parameter_factory<ExecutionAgent>() a name
template<class ExecutionAgent, class BarrierOrVoid>
using agent_shared_parameter_factory = decltype(
  make_agent_shared_parameter_factory<ExecutionAgent>(
    std::declval<typename agency::execution_agent_traits<ExecutionAgent>::param_type>(),
    std::declval<agency::experimental::in_place_type_t<BarrierOrVoid>>()
  )
);


} // end detail
} // end agency

