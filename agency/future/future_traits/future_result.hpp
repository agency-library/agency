#pragma once

#include <agency/detail/config.hpp>
#include <utility>

namespace agency
{


template<class Future>
struct future_result
{
  // the decay removes the reference returned
  // from futures like shared_future
  // the idea is given Future<T>,
  // future_result<Future<T>> returns T
  using type = typename std::decay<
    decltype(std::declval<Future>().get())
  >::type;
};


template<class Future>
using future_result_t = typename future_result<Future>::type;


} // end agency

