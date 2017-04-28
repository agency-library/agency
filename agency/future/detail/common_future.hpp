#pragma once

#include <agency/detail/config.hpp>
#include <agency/future/variant_future.hpp>

namespace agency
{
namespace detail
{
namespace common_future_detail
{


// common_future2 is a type trait which takes two Future
// types and returns a future type which can represent either

template<class Future1, class Future2>
struct common_future2
{
  // when the Future types are different, the result is variant_future
  using type = variant_future<Future1,Future2>;
};

template<class Future>
struct common_future2<Future,Future>
{
  // when the Future types are the same type, the result is that type
  using type = Future;
};

template<class Future1, class Future2>
using common_future2_t = typename common_future2<Future1,Future2>::type;


} // end common_future_detail


// common_future is a type trait which takes many Future types
// and returns a future type which can represent any of them
template<class Future, class... Futures>
struct common_future;

template<class Future, class... Futures>
using common_future_t = typename common_future<Future,Futures...>::type;

// Recursive case
template<class Future1, class Future2, class... Futures>
struct common_future<Future1,Future2,Futures...>
{
  using type = common_future_t<Future1, common_future_t<Future2, Futures...>>;
};

// base case 1: a single Future
template<class Future>
struct common_future<Future>
{
  using type = Future;
};

// base case 2: two Futures
template<class Future1, class Future2>
struct common_future<Future1,Future2>
{
  using type = common_future_detail::common_future2_t<Future1,Future2>;
};


} // end detail
} // end agency

