#pragma once

#include <agency/detail/config.hpp>
#include <agency/future/variant_future.hpp>

namespace agency
{
namespace detail
{
namespace future_sum_detail
{


// future_sum2 is a type trait which takes two Future
// types and returns a future type which can represent either

template<class Future1, class Future2>
struct future_sum2
{
  // when the Future types are different, the result is variant_future
  using type = variant_future<Future1,Future2>;
};

template<class Future>
struct future_sum2<Future,Future>
{
  // when the Future types are the same type, the result is that type
  using type = Future;
};

template<class Future1, class Future2>
using future_sum2_t = typename future_sum2<Future1,Future2>::type;


} // end future_sum_detail


// future_sum is a type trait which takes many Future types
// and returns a future type which can represent any of them
// it is a "sum type" for Futures
template<class Future, class... Futures>
struct future_sum;

template<class Future, class... Futures>
using future_sum_t = typename future_sum<Future,Futures...>::type;

// Recursive case
template<class Future1, class Future2, class... Futures>
struct future_sum<Future1,Future2,Futures...>
{
  using type = future_sum_t<Future1, future_sum_t<Future2, Futures...>>;
};

// base case 1: a single Future
template<class Future>
struct future_sum<Future>
{
  using type = Future;
};

// base case 2: two Futures
template<class Future1, class Future2>
struct future_sum<Future1,Future2>
{
  using type = future_sum_detail::future_sum2_t<Future1,Future2>;
};


} // end detail
} // end agency

