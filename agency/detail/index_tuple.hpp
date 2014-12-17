#pragma once

#include <agency/detail/tuple.hpp>
#include <agency/detail/arithmetic_tuple_facade.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{


template<class... Indices>
class index_tuple :
  public agency::detail::tuple<Indices...>,
  public arithmetic_tuple_facade<index_tuple<Indices...>>
{
  public:
    using agency::detail::tuple<Indices...>::tuple;
};


template<class... Indices>
__AGENCY_ANNOTATION
index_tuple<Indices...> make_index_tuple(const std::tuple<Indices...>& indices)
{
  return index_tuple<Indices...>(indices);
}

template<class... Args>
__AGENCY_ANNOTATION
index_tuple<decay_t<Args>...> make_index_tuple(Args&&... args)
{
  return index_tuple<decay_t<Args>...>(std::forward<Args>(args)...);
}


struct index_tuple_maker
{
  template<class... Args>
  __AGENCY_ANNOTATION
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


namespace __tu
{

// tuple_traits specializations

template<class... Indices>
struct tuple_traits<agency::detail::index_tuple<Indices...>>
  : __tu::tuple_traits<agency::detail::tuple<Indices...>>
{
  using tuple_type = agency::detail::tuple<Indices...>;
}; // end tuple_traits


} // end __tu


namespace std
{


template<class... Indices>
struct tuple_size<agency::detail::index_tuple<Indices...>> : std::tuple_size<agency::detail::tuple<Indices...>> {};

template<size_t i, class... Indices>
struct tuple_element<i,agency::detail::index_tuple<Indices...>> : std::tuple_element<i,agency::detail::tuple<Indices...>> {};


} // end namespace std

