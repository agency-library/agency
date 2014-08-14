#pragma once

#include <integer_sequence>
#include <type_traits>
#include <tuple>

template<class Head, class... Tail, size_t... I>
std::tuple<Tail...> __tuple_tail(const std::tuple<Head,Tail...> &t, std::index_sequence<I...>)
{
  return std::make_tuple(std::get<I+1>(t)...);
}

template<class Head, class... Tail>
std::tuple<Tail...> tuple_tail(const std::tuple<Head,Tail...> &t)
{
  typedef std::tuple<Head,Tail...> tuple_type;
  using indices = std::make_index_sequence<std::tuple_size<typename std::decay<tuple_type>::type>::value - 1>;
  return __tuple_tail(t, indices{});
}

template<class T1, class T2>
std::tuple<T2> tuple_tail(const std::pair<T1,T2> &p)
{
  return std::make_tuple(p.second);
}

