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


template<class Future, bool Enable = has_value_type<Future>::value>
struct future_value
{
  using type = typename Future::value_type;
};


template<template<class> class Future, class T>
struct future_value<Future<T>, false>
{
  using type = T;
};


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
  static rebind<T> make_ready(T&& value)
  {
    return rebind<T>::make_ready();
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
  static rebind<U> make_ready(U&& value)
  {
    return detail::make_ready_future(std::forward<U>(value));
  }
};


} // end agency

