#pragma once

#include <tuple>
#include <agency/detail/integer_sequence.hpp>

namespace agency
{
namespace detail
{


template<class Tuple, size_t... I>
auto tuple_of_references_impl(Tuple& t, agency::detail::index_sequence<I...>)
  -> decltype(std::tie(std::get<I>(t)...))
{
  return std::tie(std::get<I>(t)...);
}

template<class Tuple>
auto tuple_of_references(Tuple& t)
  -> decltype(
       tuple_of_references_impl(
         t,
         agency::detail::make_index_sequence<
           std::tuple_size<
             typename std::decay<Tuple>::type
           >::value
         >()
       )
     )
{
  return tuple_of_references_impl(
    t,
    agency::detail::make_index_sequence<
      std::tuple_size<
        typename std::decay<Tuple>::type
      >::value
    >()
  );
}

// XXX dunno why the following doesn't work:
//#include <__tuple_map>
//
//struct __forward
//{
//  template<class T>
//  auto operator()(T&& x)
//    -> decltype(std::forward<T>(x))
//  {
//    return std::forward<T>(x);
//  }
//};
//
//template<class Tuple>
//auto tuple_of_references(Tuple&& t)
//  -> decltype(__tuple_map(std::forward<Tuple>(t), __forward()))
//{
//  return __tuple_map(std::forward<Tuple>(t), __forward());
//}


} // end detail
} // end agency

