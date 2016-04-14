#pragma once

#include <agency/detail/bulk_invoke/bind_agent_local_parameters.hpp>
#include <agency//detail/bulk_invoke/bind.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <utility>

namespace agency
{
namespace cuda
{
namespace detail
{


// if J... is a bit vector indicating which elements of args are shared parameters
// then I... is the exclusive scan of J
template<size_t... I, class Function, class... Args>
__AGENCY_ANNOTATION
auto bind_agent_local_parameters_impl(agency::detail::index_sequence<I...>, Function f, Args&&... args)
  -> decltype(
       agency::detail::bind(f, agency::detail::hold_shared_parameters_place<1 + I>(std::forward<Args>(args))...)
     )
{
  // we add 1 to I to account for the executor_idx argument which will be passed as the first parameter to f
  return agency::detail::bind(f, agency::detail::hold_shared_parameters_place<1 + I>(std::forward<Args>(args))...);
}


template<class Function, class... Args>
auto bind_agent_local_parameters(Function f, Args&&... args)
  -> decltype(
       detail::bind_agent_local_parameters_impl(agency::detail::scanned_shared_argument_indices<Args...>{}, f, std::forward<Args>(args)...)
     )
{
  return detail::bind_agent_local_parameters_impl(agency::detail::scanned_shared_argument_indices<Args...>{}, f, std::forward<Args>(args)...);
}


// if J... is a bit vector indicating which elements of args are shared parameters
// then I... is the exclusive scan of J
template<size_t index_of_first_agent_local_parameter, size_t... I, class Function, class... Args>
__AGENCY_ANNOTATION
auto new_bind_agent_local_parameters_impl(agency::detail::index_sequence<I...>, Function f, Args&&... args)
  -> decltype(
       agency::detail::bind(f, agency::detail::hold_shared_parameters_place<index_of_first_agent_local_parameter + I>(std::forward<Args>(args))...)
     )
{
  // we add 1 to I to account for the executor_idx argument which will be passed as the first parameter to f
  return agency::detail::bind(f, agency::detail::hold_shared_parameters_place<index_of_first_agent_local_parameter + I>(std::forward<Args>(args))...);
}


template<size_t index_of_first_agent_local_parameter, class Function, class... Args>
auto new_bind_agent_local_parameters(std::integral_constant<size_t,index_of_first_agent_local_parameter>, Function f, Args&&... args)
  -> decltype(
       detail::bind_agent_local_parameters_impl(agency::detail::scanned_shared_argument_indices<Args...>{}, f, std::forward<Args>(args)...)
     )
{
  return detail::new_bind_agent_local_parameters_impl<index_of_first_agent_local_parameter>(agency::detail::scanned_shared_argument_indices<Args...>{}, f, std::forward<Args>(args)...);
}


} // end detail
} // end cuda
} // end agency

