#pragma once

#include <agency/detail/config.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace allocator_traits_detail
{


template<class Alloc, class T, class... Args>
struct has_construct_impl
{
  template<
    class Alloc1,
    class = decltype(
      std::declval<Alloc1*>()->construct(
        std::declval<T*>(),
        std::declval<Args>()...
      )
    )
  >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Alloc>(0));
};

template<class Alloc, class T, class... Args>
using has_construct = typename has_construct_impl<Alloc,T*,Args...>::type;


template<class Alloc, class Iterator, class... Iterators>
struct has_construct_n_impl
{
  template<
    class Alloc1,
    class Result = decltype(
      std::declval<Alloc1&>().construct_n(
        std::declval<Iterator>(),
        std::declval<size_t>(),
        std::declval<Iterators>()...
      )
    ),
    class = typename std::enable_if<
      std::is_convertible<Result,Iterator>::value
    >::type
  >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Alloc>(0));
};

template<class Alloc, class Iterator, class... Iterators>
using has_construct_n = typename has_construct_n_impl<Alloc,Iterator,Iterators...>::type;


} // end allocator_traits_detail
} // end detail
} // end agency

