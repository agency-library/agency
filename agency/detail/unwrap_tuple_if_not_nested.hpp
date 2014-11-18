#pragma once

#include <agency/execution_categories.hpp>
#include <tuple>
#include <utility>

namespace agency
{
namespace detail
{


template<class ExecutionCategory, class Tuple>
static auto unwrap_tuple_if_not_nested_impl(ExecutionCategory, Tuple&& t)
  -> decltype(std::get<0>(std::forward<Tuple>(t)))
{
  return std::get<0>(std::forward<Tuple>(t));
}

template<class ExecutionCategory1, class ExecutionCategory2, class Tuple>
static auto unwrap_tuple_if_not_nested_impl(nested_execution_tag<ExecutionCategory1,ExecutionCategory2>, Tuple&& t)
  -> decltype(std::forward<Tuple>(t))
{
  return std::forward<Tuple>(t);
}


template<class ExecutionCategory, class Tuple>
auto unwrap_tuple_if_not_nested(Tuple&& t)
  -> decltype(
       unwrap_tuple_if_not_nested_impl(ExecutionCategory(), std::forward<Tuple>(t))
     )
{
  return unwrap_tuple_if_not_nested_impl(ExecutionCategory(), std::forward<Tuple>(t));
}


} // end detail
} // end agency

