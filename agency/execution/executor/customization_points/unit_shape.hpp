#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/coordinate/detail/shape/shape_cast.hpp>
#include <type_traits>


namespace agency
{
namespace detail
{


template<class Executor, class Shape>
struct has_unit_shape_impl
{
  template<
    class Executor1,
    class ReturnType = decltype(std::declval<Executor1>().unit_shape()),
    __AGENCY_REQUIRES(
      std::is_same<ReturnType,Shape>::value
    )
  >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};


template<class Executor, class Shape>
using has_unit_shape = typename has_unit_shape_impl<Executor,Shape>::type;


} // end detail


// this overload handles the case when an Executor has .unit_shape()
__agency_exec_check_disable__
template<class E,
         __AGENCY_REQUIRES(is_executor<E>::value),
         __AGENCY_REQUIRES(detail::has_unit_shape<E,executor_shape_t<E>>::value)
        >
__AGENCY_ANNOTATION
executor_shape_t<E> unit_shape(const E& exec)
{
  return exec.unit_shape();
}


// this overload handles the case when an Executor does not have .unit_shape()
__agency_exec_check_disable__
template<class E,
         __AGENCY_REQUIRES(is_executor<E>::value),
         __AGENCY_REQUIRES(!detail::has_unit_shape<E,executor_shape_t<E>>::value)
        >
__AGENCY_ANNOTATION
executor_shape_t<E> unit_shape(const E&)
{
  // by default, an executor's unit shape contains a single point
  return detail::shape_cast<executor_shape_t<E>>(1);
}


} // end agency

