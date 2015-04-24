#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/exception_list.hpp>
#include <future>
#include <utility>

namespace agency
{
namespace detail
{


template<class T>
std::future<decay_t<T>> make_ready_future(T&& value)
{
  std::promise<decay_t<T>> p;
  p.set_value(std::forward<T>(value));
  return p.get_future();
}


inline std::future<void> make_ready_future()
{
  std::promise<void> p;
  p.set_value();
  return p.get_future();
}


// XXX when_all is supposed to return a future<vector>
template<typename ForwardIterator>
std::future<void> when_all(ForwardIterator first, ForwardIterator last)
{
  exception_list exceptions = flatten_into_exception_list(first, last);

  std::promise<void> p;

  if(exceptions.size() > 0)
  {
    p.set_exception(std::make_exception_ptr(exceptions));
  }
  else
  {
    p.set_value();
  }

  return p.get_future();
}


template<class T, class Function>
std::future<typename std::result_of<Function(std::future<T>&)>::type>
  then(std::future<T>& fut, std::launch policy, Function&& f)
{
  return std::async(policy, [](std::future<T>&& fut, Function&& f)
  {
    fut.wait();
    return std::forward<Function>(f)(fut);
  },
  std::move(fut),
  std::forward<Function>(f)
  );
}


template<class T, class Function>
std::future<typename std::result_of<Function(std::future<T>&)>::type>
  then(std::future<T>& fut, Function&& f)
{
  return detail::then(fut, std::launch::async | std::launch::deferred, std::forward<Function>(f));
}


// XXX should check for nested ::rebind<T> i guess
template<class Future, class T>
struct rebind_future_value;


template<template<class> class Future, class FromType, class ToType>
struct rebind_future_value<Future<FromType>,ToType>
{
  using type = Future<ToType>;
};


__DEFINE_HAS_NESTED_TYPE(has_value_type, value_type);


template<class Future>
struct future_value
{
  using type = decltype(std::declval<Future>().get());
};


namespace is_future_detail
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


} // end is_future_detail


template<class T>
struct is_future
  : std::integral_constant<
      bool,
      is_future_detail::has_wait<T>::value && is_future_detail::has_get<T>::value
    >
{};


template<class T, template<class> class Future, class Enable = void>
struct is_instance_of_future : std::false_type {};

template<class T, template<class> class Future>
struct is_instance_of_future<T,Future,
  typename std::enable_if<
    is_future<T>::value
  >::type
> : std::is_same<
  T,
  Future<
    typename future_value<T>::type
  >
>
{};


} // end detail


template<class Future>
struct future_traits
{
  using future_type = Future;

  using value_type = typename detail::future_value<future_type>::type;

  template<class U>
  using rebind = typename detail::rebind_future_value<future_type,U>::type;

  __AGENCY_ANNOTATION
  static rebind<void> make_ready()
  {
    return rebind<void>::make_ready();
  }

  template<class T>
  __AGENCY_ANNOTATION
  static rebind<typename std::decay<T>::type> make_ready(T&& value)
  {
    return rebind<typename std::decay<T>::type>::make_ready(std::forward<T>(value));
  }

  private:
  template<class Future1>
  struct has_discard_value
  {
    template<class Future2,
             class = decltype(std::declval<Future2*>->discard_value())
            >
    static std::true_type test(int);

    template<class>
    static std::false_type test(...);

    using type = decltype(test<Future1>);
  };

  static rebind<void> discard_value(future_type& fut, std::true_type)
  {
    return fut.discard_value();
  }

  public:

  __AGENCY_ANNOTATION
  static rebind<void> discard_value(future_type& fut)
  {
    return future_traits::discard_value(fut, typename has_discard_value<future_type>::type());
  }
};


template<class T>
struct future_traits<std::future<T>>
{
  using future_type = std::future<T>;

  using value_type = typename detail::future_value<future_type>::type;

  template<class U>
  using rebind = typename detail::rebind_future_value<future_type,U>::type;

  static rebind<void> make_ready()
  {
    return detail::make_ready_future();
  }

  template<class U>
  static rebind<typename std::decay<U>::type> make_ready(U&& value)
  {
    return detail::make_ready_future(std::forward<U>(value));
  }

  template<class Future,
           class = typename std::enable_if<
             std::is_same<Future,future_type>::value &&
             std::is_empty<typename future_traits<Future>::value_type>::value
           >::type>
  static std::future<void> discard_value(Future& fut)
  {
    return std::move(*reinterpret_cast<std::future<void>*>(&fut));
  }
};


} // end agency

