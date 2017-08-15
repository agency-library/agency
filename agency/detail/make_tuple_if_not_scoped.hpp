#pragma once

#include <agency/execution/execution_categories.hpp>
#include <agency/tuple.hpp>
#include <utility>


namespace agency
{
namespace detail
{


// execution is scoped, just return x
template<class ExecutionCategory1, class ExecutionCategory2, class T>
__AGENCY_ANNOTATION
T make_tuple_if_not_scoped(agency::scoped_execution_tag<ExecutionCategory1,ExecutionCategory2>, const T& x)
{
  return x;
}


// execution is not scoped, wrap up x in a tuple
template<class ExecutionCategory, class T>
__AGENCY_ANNOTATION
agency::detail::tuple<T> make_tuple_if_not_scoped(ExecutionCategory, const T& x)
{
  return agency::detail::make_tuple(x);
}


template<class ExecutionCategory, class T>
__AGENCY_ANNOTATION
auto make_tuple_if_not_scoped(const T& x)
  -> decltype(agency::detail::make_tuple_if_not_scoped(ExecutionCategory(), x))
{
  return agency::detail::make_tuple_if_not_scoped(ExecutionCategory(), x);
}


template<class ExecutionCategory1, class ExecutionCategory2, class T>
__AGENCY_ANNOTATION
auto tie_if_not_scoped(scoped_execution_tag<ExecutionCategory1,ExecutionCategory2>, T&& x)
  -> decltype(std::forward<T>(x))
{
  return std::forward<T>(x);
}


template<class ExecutionCategory, class T>
__AGENCY_ANNOTATION
auto tie_if_not_scoped(ExecutionCategory, T&& x)
  -> decltype(agency::detail::tie(std::forward<T>(x)))
{
  return agency::detail::tie(std::forward<T>(x));
}


template<class ExecutionCategory, class T>
__AGENCY_ANNOTATION
auto tie_if_not_scoped(T&& x)
  -> decltype(tie_if_not_scoped(ExecutionCategory(), std::forward<T>(x)))
{
  return tie_if_not_scoped(ExecutionCategory(), std::forward<T>(x));
}


} // end detail
} // end agency

