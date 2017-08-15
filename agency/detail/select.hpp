#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/tuple.hpp>
#include <tuple>
#include <utility>
#include <type_traits>

namespace agency
{
namespace detail
{


template<class TupleReference, size_t... Indices>
struct select_from_tuple_result
{
  using tuple_type = typename std::decay<TupleReference>::type;

  using type = tuple<
    propagate_reference_t<
      TupleReference,
      typename std::tuple_element<
        Indices,
        tuple_type
      >::type
    >...
  >;
};

template<class TupleReference, size_t... Indices>
using select_from_tuple_result_t = typename select_from_tuple_result<TupleReference,Indices...>::type;


template<size_t... Indices, class Tuple>
__AGENCY_ANNOTATION
select_from_tuple_result_t<Tuple&&,Indices...> select_from_tuple(Tuple&& t)
{
  return agency::forward_as_tuple(std::get<Indices>(std::forward<Tuple>(t))...);
}


template<class IndexSequence, class... Args>
struct select_result;


// no arguments, there must be nothing selected
template<size_t... Indices>
struct select_result<index_sequence<Indices...>>
{
  static_assert(sizeof...(Indices) == 0, "Too many arguments selected.");

  using type = void;
};


// no arguments, nothing selected
template<>
struct select_result<index_sequence<>>
{
  using type = void;
};


// nothing selected, return void
template<class... Args>
struct select_result<index_sequence<>,Args...>
{
  using type = void;
};


// something selected, one argument
template<size_t... Indices, class Arg>
struct select_result<index_sequence<Indices...>, Arg>
{
  // XXX check that each index is in bounds
  
  static_assert(sizeof...(Indices) == 1, "Too many arguments selected.");

  using type = Arg;
};


// no selection, one argument
template<class Arg>
struct select_result<index_sequence<>, Arg>
{
  using type = void;
};


template<size_t... Indices, class Arg1, class Arg2, class... Args>
struct select_result<index_sequence<Indices...>, Arg1, Arg2, Args...>
{
  // XXX check that each index is in bounds

  static_assert(sizeof...(Indices) <= 2 + sizeof...(Args), "Too many arguments selected.");

  using type = select_from_tuple_result_t<
    tuple<Arg1,Arg2,Args...>,
    Indices...
  >;
};


template<class IndexSequence, class... Args>
using select_result_t = typename select_result<IndexSequence, Args...>::type;


template<size_t... Indices>
__AGENCY_ANNOTATION
select_result_t<index_sequence<Indices...>>
  select()
{
  static_assert(sizeof...(Indices) == 0, "Too many arguments selected.");
}


template<size_t... Indices, class Arg>
__AGENCY_ANNOTATION
select_result_t<index_sequence<Indices...>, Arg&&>
  select(Arg&& arg)
{
  // XXX check that each index is in bounds
  
  static_assert(sizeof...(Indices) <= 1, "Too many arguments selected.");

  using result_type = select_result_t<index_sequence<Indices...>, Arg&&>;

  return static_cast<result_type>(std::forward<Arg>(arg));
}


template<size_t... Indices, class Arg1, class Arg2, class... Args>
__AGENCY_ANNOTATION
select_result_t<index_sequence<Indices...>, Arg1, Arg2, Args...>
  select(Arg1&& arg1, Arg2&& arg2, Args&&... args)
{
  // XXX check that each index is in bounds

  static_assert(sizeof...(Indices) <= 2 + sizeof...(Args), "Too many arguments selected.");

  return select_from_tuple<Indices...>(agency::forward_as_tuple(std::forward<Arg1>(arg1), std::forward<Arg2>(arg2), std::forward<Args>(args)...));
}


} // end detail
} // end agency

