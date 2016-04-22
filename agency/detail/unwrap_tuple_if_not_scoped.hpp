#pragma once

#include <agency/execution_categories.hpp>
#include <agency/detail/tuple.hpp>
#include <utility>

namespace agency
{
namespace detail
{


template<class ExecutionCategory, class Tuple>
__AGENCY_ANNOTATION
static auto unwrap_tuple_if_not_scoped_impl(ExecutionCategory, Tuple&& t)
  -> decltype(detail::get<0>(std::forward<Tuple>(t)))
{
  return detail::get<0>(std::forward<Tuple>(t));
}

template<class ExecutionCategory1, class ExecutionCategory2, class Tuple>
__AGENCY_ANNOTATION
static auto unwrap_tuple_if_not_scoped_impl(scoped_execution_tag<ExecutionCategory1,ExecutionCategory2>, Tuple&& t)
  -> decltype(std::forward<Tuple>(t))
{
  return std::forward<Tuple>(t);
}


template<class ExecutionCategory, class Tuple>
__AGENCY_ANNOTATION
auto unwrap_tuple_if_not_scoped(Tuple&& t)
  -> decltype(
       unwrap_tuple_if_not_scoped_impl(ExecutionCategory(), std::forward<Tuple>(t))
     )
{
  return unwrap_tuple_if_not_scoped_impl(ExecutionCategory(), std::forward<Tuple>(t));
}


} // end detail
} // end agency

