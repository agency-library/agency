#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <utility>
#include <type_traits>

namespace agency
{
namespace detail
{


template<class T>
struct has_wait_impl
{
  template<class Future,
           class = decltype(
             std::declval<Future>().wait()
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<T>(0));
};


template<class T>
using has_wait = typename has_wait_impl<T>::type;


template<class T>
struct has_get_impl
{
  template<class Future,
           class = decltype(std::declval<Future>().get())
           >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<T>(0));
};


template<class T>
using has_get = typename has_get_impl<T>::type;


} // end detail


// XXX should probably also check for make_ready() and possibly then()
template<class T>
struct is_future : detail::conjunction<detail::has_wait<T>, detail::has_get<T>> {};


} // end agency

