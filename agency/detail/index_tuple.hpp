#pragma once

#include <agency/detail/arithmetic_tuple_facade.hpp>
#include <tuple>
#include <type_traits>

namespace agency
{
namespace detail
{


template<class... Indices>
class index_tuple :
  public std::tuple<Indices...>,
  public arithmetic_tuple_facade<index_tuple<Indices...>>
{
  public:
    using std::tuple<Indices...>::tuple;
};


template<class... Indices>
index_tuple<Indices...> make_index_tuple(const std::tuple<Indices...>& indices)
{
  return index_tuple<Indices...>(indices);
}

template<class... Args>
index_tuple<decay_t<Args>...> make_index_tuple(Args&&... args)
{
  return index_tuple<decay_t<Args>...>(std::forward<Args>(args)...);
}


struct index_tuple_maker
{
  template<class... Args>
  auto operator()(Args&&... args) const
    -> decltype(
         make_index_tuple(std::forward<Args>(args)...)
       )
  {
    return make_index_tuple(std::forward<Args>(args)...);
  }
};


} // end detail
} // end agency


namespace std
{


template<class... Indices>
struct tuple_size<agency::detail::index_tuple<Indices...>> : std::tuple_size<std::tuple<Indices...>> {};

template<size_t i, class... Indices>
struct tuple_element<i,agency::detail::index_tuple<Indices...>> : std::tuple_element<i,std::tuple<Indices...>> {};

template<size_t i, class... Indices>
auto get(agency::detail::index_tuple<Indices...>& t)
  -> decltype(
       std::get<i>(static_cast<std::tuple<Indices...>&>(t))
     )
{
  return std::get<i>(static_cast<std::tuple<Indices...>&>(t));
}


template<size_t i, class... Indices>
auto get(const agency::detail::index_tuple<Indices...>& t)
  -> decltype(
       std::get<i>(static_cast<const std::tuple<Indices...>&>(t))
     )
{
  return std::get<i>(static_cast<const std::tuple<Indices...>&>(t));
}


template<size_t i, class... Indices>
auto get(agency::detail::index_tuple<Indices...>&& t)
  -> decltype(
       std::get<i>(std::move<std::tuple<Indices...>&&>(t))
     )
{
  return std::get<i>(std::move<std::tuple<Indices...>&&>(t));
}


} // end namespace std

