#pragma once

#include <agency/detail/config.hpp>
#include <agency/memory/allocator/variant_allocator.hpp>

namespace agency
{
namespace detail
{
namespace allocator_sum_detail
{


// allocator_sum2 is a type trait which takes two Allocator
// types and returns a allocator type which can represent either

template<class Allocator1, class Allocator2>
struct allocator_sum2
{
  // when the Allocator types are different, the result is variant_allocator
  using type = variant_allocator<Allocator1,Allocator2>;
};

template<class Allocator>
struct allocator_sum2<Allocator,Allocator>
{
  // when the Allocator types are the same type, the result is that type
  using type = Allocator;
};

template<class Allocator1, class Allocator2>
using allocator_sum2_t = typename allocator_sum2<Allocator1,Allocator2>::type;


} // end allocator_sum_detail


// allocator_sum is a type trait which takes many Future types
// and returns a allocator type which can represent any of them
// it is a "sum type" for Allocators
template<class Allocator, class... Allocators>
struct allocator_sum;

template<class Allocator, class... Allocators>
using allocator_sum_t = typename allocator_sum<Allocator,Allocators...>::type;

// Recursive case
template<class Allocator1, class Allocator2, class... Allocators>
struct allocator_sum<Allocator1,Allocator2,Allocators...>
{
  using type = allocator_sum_t<Allocator1, allocator_sum_t<Allocator2, Allocators...>>;
};

// base case 1: a single Allocator
template<class Allocator>
struct allocator_sum<Allocator>
{
  using type = Allocator;
};

// base case 2: two Allocators
template<class Allocator1, class Allocator2>
struct allocator_sum<Allocator1,Allocator2>
{
  using type = allocator_sum_detail::allocator_sum2_t<Allocator1,Allocator2>;
};


} // end detail
} // end agency

