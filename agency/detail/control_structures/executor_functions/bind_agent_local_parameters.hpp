#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/control_structures/bind.hpp>
#include <agency/detail/control_structures/shared_parameter.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{

template<size_t I, class Arg>
__AGENCY_ANNOTATION
typename std::enable_if<
  !is_shared_parameter<typename std::decay<Arg>::type>::value,
  Arg&&
>::type
  hold_shared_parameters_place(Arg&& arg)
{
  return std::forward<Arg>(arg);
}


template<size_t I, class T>
__AGENCY_ANNOTATION
typename std::enable_if<
  is_shared_parameter<typename std::decay<T>::type>::value,
  placeholder<I>
>::type
  hold_shared_parameters_place(T&&)
{
  return placeholder<I>{};
}


// if J... is a bit vector indicating which elements of args are shared parameters
// then I... is the exclusive scan of J
template<size_t index_of_first_user_parameter, size_t... I, class Function, class... Args>
__AGENCY_ANNOTATION
auto bind_agent_local_parameters_impl(index_sequence<I...>, Function f, Args&&... args)
  -> decltype(
       detail::bind(f, hold_shared_parameters_place<index_of_first_user_parameter + I>(std::forward<Args>(args))...)
     )
{
  return detail::bind(f, hold_shared_parameters_place<index_of_first_user_parameter + I>(std::forward<Args>(args))...);
}


template<class... Args>
struct arg_is_shared
{
  using tuple_type = tuple<Args...>;

  template<size_t i>
  using map = is_shared_parameter<
    typename std::decay<
      typename std::tuple_element<i,tuple_type>::type
    >::type
  >;
};


// XXX nvcc 7.0 doesnt like agency::detail::scanned_shared_argument_indices
//     so WAR it by implementing a slightly different version here
template<class... Args>
struct scanned_shared_argument_indices_impl
{
  using type = agency::detail::transform_exclusive_scan_index_sequence<
    agency::detail::arg_is_shared<Args...>::template map,
    0,
    // XXX various compilers have difficulty with index_sequence_for, so WAR it
    //index_sequence_for<Args...>
    agency::detail::make_index_sequence<sizeof...(Args)>
  >;
};


template<class... Args>
using scanned_shared_argument_indices = typename scanned_shared_argument_indices_impl<Args...>::type;


template<size_t index_of_first_agent_local_parameter, class Function, class... Args>
__AGENCY_ANNOTATION
auto bind_agent_local_parameters(Function f, Args&&... args)
  -> decltype(
       bind_agent_local_parameters_impl<index_of_first_agent_local_parameter>(scanned_shared_argument_indices<Args...>{}, f, std::forward<Args>(args)...)
     )
{
  return bind_agent_local_parameters_impl<index_of_first_agent_local_parameter>(scanned_shared_argument_indices<Args...>{}, f, std::forward<Args>(args)...);
}


// XXX this function worksaround nvbug 1754712
// XXX eliminate it when that bug no longer exists
template<size_t index_of_first_agent_local_parameter, class Function, class... Args>
__AGENCY_ANNOTATION
auto bind_agent_local_parameters_workaround_nvbug1754712(std::integral_constant<size_t,index_of_first_agent_local_parameter>, Function f, Args&&... args)
  -> decltype(
       bind_agent_local_parameters_impl<index_of_first_agent_local_parameter>(scanned_shared_argument_indices<Args...>{}, f, std::forward<Args>(args)...)
     )
{
  return bind_agent_local_parameters_impl<index_of_first_agent_local_parameter>(scanned_shared_argument_indices<Args...>{}, f, std::forward<Args>(args)...);
}


} // end detail
} // end agency

