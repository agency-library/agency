#pragma once

#include <agency/detail/config.hpp>
#include <utility>

namespace agency
{


template<class Future>
struct future_value
{
  // the decay removes the reference returned
  // from futures like shared_future
  // the idea is given Future<T>,
  // future_value<Future<T>> returns T
  using type = typename std::decay<
    decltype(std::declval<Future>().get())
  >::type;
};


template<class Future>
using future_value_t = typename future_value<Future>::type;


} // end agency

