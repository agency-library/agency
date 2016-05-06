#pragma once

#include <agency/detail/config.hpp>
#include <agency/executor/executor_traits.hpp>
#include <agency/executor/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/shape.hpp>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
{


template<class Executor>
__AGENCY_ANNOTATION
typename std::enable_if<
  !has_max_shape_dimensions<Executor>::value,
  typename executor_traits<Executor>::shape_type
>::type
  max_shape_dimensions(const Executor&)
{
  return detail::max_shape_dimensions<typename executor_traits<Executor>::shape_type>();
} // end executor_type::max_shape_dimensions()


__agency_exec_check_disable__
template<class Executor>
__AGENCY_ANNOTATION
typename std::enable_if<
  has_max_shape_dimensions<Executor>::value,
  typename executor_traits<Executor>::shape_type
>::type
  max_shape_dimensions(const Executor& ex)
{
  return ex.max_shape_dimensions();
} // end executor_type::max_shape_dimensions()


} // end executor_traits_detail
} // end detail


template<class Executor>
__AGENCY_ANNOTATION
typename executor_traits<Executor>::shape_type
  executor_traits<Executor>
    ::max_shape_dimensions(const typename executor_traits<Executor>::executor_type& ex)
{
  return detail::executor_traits_detail::max_shape_dimensions(ex);
} // end executor_traits::max_shape_dimensions()


} // end agency

