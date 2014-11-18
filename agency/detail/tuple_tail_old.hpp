#pragma once

#include <integer_sequence>
#include <type_traits>
#include <tuple>

// XXX eliminate this function, which is redundant
//     with the more general version in tuple_utility.hpp

namespace agency
{
namespace detail
{


template<class Head, class... Tail, size_t... I>
std::tuple<Tail...> tuple_tail_old_impl(const std::tuple<Head,Tail...> &t, std::index_sequence<I...>)
{
  return std::tie(std::get<I+1>(t)...);
}

template<class Head, class... Tail>
std::tuple<Tail...> tuple_tail_old(const std::tuple<Head,Tail...> &t)
{
  typedef std::tuple<Head,Tail...> tuple_type;
  using indices = std::make_index_sequence<std::tuple_size<typename std::decay<tuple_type>::type>::value - 1>;
  return tuple_tail_old_impl(t, indices{});
}

template<class T1, class T2>
std::tuple<T2> tuple_tail_old(const std::pair<T1,T2> &p)
{
  return std::make_tuple(p.second);
}


} // end detail
} // end agency

