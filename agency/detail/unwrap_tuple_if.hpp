#pragma once

#include <agency/tuple.hpp>
#include <utility>
#include <type_traits>

namespace agency
{
namespace detail
{


template<class Tuple>
__AGENCY_ANNOTATION
static auto unwrap_tuple_if_impl(std::true_type, Tuple&& t)
  -> decltype(agency::get<0>(std::forward<Tuple>(t)))
{
  return agency::get<0>(std::forward<Tuple>(t));
}

template<class Tuple>
__AGENCY_ANNOTATION
static auto unwrap_tuple_if_impl(std::false_type, Tuple&& t)
  -> decltype(std::forward<Tuple>(t))
{
  return std::forward<Tuple>(t);
}


template<bool b, class Tuple>
__AGENCY_ANNOTATION
auto unwrap_tuple_if(Tuple&& t)
  -> decltype(
       detail::unwrap_tuple_if_impl(std::integral_constant<bool,b>(), std::forward<Tuple>(t))
     )
{
  return detail::unwrap_tuple_if_impl(std::integral_constant<bool,b>(), std::forward<Tuple>(t));
}


} // end detail
} // end agency

