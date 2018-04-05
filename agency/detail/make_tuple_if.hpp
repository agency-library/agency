#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/tuple.hpp>
#include <utility>
#include <type_traits>


namespace agency
{
namespace detail
{


template<class T>
__AGENCY_ANNOTATION
agency::tuple<T> make_tuple_if_impl(std::true_type, const T& x)
{
  return agency::make_tuple(x);
}


template<class T>
__AGENCY_ANNOTATION
T make_tuple_if_impl(std::false_type, const T& x)
{
  return x;
}


template<bool b, class T>
__AGENCY_ANNOTATION
auto make_tuple_if(const T& x)
  -> decltype(agency::detail::make_tuple_if_impl(std::integral_constant<bool, b>(), x))
{
  return agency::detail::make_tuple_if_impl(std::integral_constant<bool, b>(), x);
}



template<class T>
__AGENCY_ANNOTATION
auto tie_if(std::false_type, T&& x)
  -> decltype(std::forward<T>(x))
{
  return std::forward<T>(x);
}


template<class T>
__AGENCY_ANNOTATION
auto tie_if(std::true_type, T&& x)
  -> decltype(agency::tie(std::forward<T>(x)))
{
  return agency::tie(std::forward<T>(x));
}


template<bool b, class T>
__AGENCY_ANNOTATION
auto tie_if(T&& x)
  -> decltype(tie_if(std::integral_constant<bool, b>(), std::forward<T>(x)))
{
  return tie_if(std::integral_constant<bool, b>(), std::forward<T>(x));
}


} // end detail
} // end agency

