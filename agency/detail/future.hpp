#pragma once

#include <future>
#include <type_traits>
#include <agency/exception_list.hpp>

namespace agency
{
namespace detail
{


template<class T>
std::future<std::decay_t<T>> make_ready_future(T&& value)
{
  std::promise<std::decay_t<T>> p;
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


} // end detail
} // end agency

