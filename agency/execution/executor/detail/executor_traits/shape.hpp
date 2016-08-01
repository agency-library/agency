#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/executor/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/shape_cast.hpp>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
{


template<class Executor>
__AGENCY_ANNOTATION
typename std::enable_if<
  !has_shape<Executor>::value,
  typename executor_traits<Executor>::shape_type
>::type
  shape(const Executor&)
{
  return detail::shape_cast<typename executor_traits<Executor>::shape_type>(1);
} // end executor_traits::shape()


__agency_exec_check_disable__
template<class Executor>
__AGENCY_ANNOTATION
typename std::enable_if<
  has_shape<Executor>::value,
  typename executor_traits<Executor>::shape_type
>::type
  shape(const Executor& ex)
{
  return ex.shape();
} // end executor_traits::shape()


} // end executor_traits_detail
} // end detail


template<class Executor>
__AGENCY_ANNOTATION
typename executor_traits<Executor>::shape_type
  executor_traits<Executor>
    ::shape(const typename executor_traits<Executor>::executor_type& ex)
{
  return detail::executor_traits_detail::shape(ex);
} // end executor_traits::shape()


} // end agency

