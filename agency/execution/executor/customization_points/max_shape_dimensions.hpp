#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/shape.hpp>
#include <agency/execution/executor/executor_traits/executor_shape.hpp>
#include <utility>
#include <type_traits>


namespace agency
{
namespace detail
{


template<class Executor, class Shape>
struct has_max_shape_dimensions_impl
{
  template<
    class Executor1,
    class ReturnType = decltype(std::declval<Executor1>().max_shape_dimensions()),
    __AGENCY_REQUIRES(std::is_same<ReturnType,Shape>::value)
  >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};


template<class Executor, class Shape>
using has_max_shape_dimensions = typename has_max_shape_dimensions_impl<Executor,Shape>::type;


} // end detail


// this overload handles the case when an Executor has .max_shape_dimensions()
__agency_exec_check_disable__
template<class E,
         __AGENCY_REQUIRES(is_executor<E>::value),
         __AGENCY_REQUIRES(detail::has_max_shape_dimensions<E,executor_shape_t<E>>::value)
        >
__AGENCY_ANNOTATION
executor_shape_t<E> max_shape_dimensions(const E& exec)
{
  return exec.max_shape_dimensions();
}


// this overload handles the case when an Executor does not have .max_shape_dimensions()
template<class E,
         __AGENCY_REQUIRES(is_executor<E>::value),
         __AGENCY_REQUIRES(!detail::has_max_shape_dimensions<E,executor_shape_t<E>>::value)
        >
__AGENCY_ANNOTATION
executor_shape_t<E> max_shape_dimensions(const E&)
{
  return detail::max_shape_dimensions<executor_shape_t<E>>();
}


} // end agency

