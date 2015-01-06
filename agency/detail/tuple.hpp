#pragma once

#include <agency/detail/config.hpp>

#define __TUPLE_ANNOTATION __AGENCY_ANNOTATION

#define __TUPLE_NAMESPACE __tu

#include <agency/detail/tuple_impl.hpp>
#include <agency/detail/tuple_utility.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/host_device_cast.hpp>
#include <utility>
#include <type_traits>

namespace agency
{
namespace detail
{


template<class... Types>
using tuple = __tu::tuple<Types...>;
using __tu::swap;
using __tu::make_tuple;
using __tu::tie;


template<class IndexSequence, class... Tuples>
struct tuple_cat_result_impl_impl;


template<size_t... I, class... Tuples>
struct tuple_cat_result_impl_impl<index_sequence<I...>, Tuples...>
{
  using type = tuple<typename __tu::__tuple_cat_get_result<I, Tuples...>::type...>;
};


template<class... Tuples>
struct tuple_cat_result_impl
{
  static const size_t result_size = __tu::__sum<0u, std::tuple_size<Tuples>::value...>::value;

  using type = typename tuple_cat_result_impl_impl<
    make_index_sequence<result_size>,
    Tuples...
  >::type;
};


template<class... Tuples>
using tuple_cat_result = typename tuple_cat_result_impl<typename std::decay<Tuples>::type...>::type;


template<class Tuple>
struct tuple_maker
{
  template<class... Args>
  __AGENCY_ANNOTATION
  Tuple operator()(Args&&... args)
  {
    return Tuple{std::forward<Args>(args)...};
  }
};


// XXX this doesn't forward tuple elements which are reference types correctly
//     because make_tuple() doesn't do that
template<class... Tuples>
__AGENCY_ANNOTATION
tuple_cat_result<Tuples...> tuple_cat(Tuples&&... tuples)
{
  return __tu::tuple_cat_apply(tuple_maker<tuple_cat_result<Tuples...>>{}, std::forward<Tuples>(tuples)...);
}


// fancy version of std::get which uses tuple_traits and can get() from things which aren't in std::
template<size_t i, class Tuple>
__AGENCY_ANNOTATION
auto get(Tuple&& t)
  -> decltype(
       __tu::tuple_traits<typename std::decay<Tuple>::type>::template get<i>(std::forward<Tuple>(t))
     )
{
  return __tu::tuple_traits<typename std::decay<Tuple>::type>::template get<i>(std::forward<Tuple>(t));
}


template<class... Args>
__AGENCY_ANNOTATION
tuple<Args&&...> forward_as_tuple(Args&&... args)
{
  return detail::tuple<Args&&...>{std::forward<Args>(args)...};
}


struct forwarder
{
  template<class... Args>
  __AGENCY_ANNOTATION
  auto operator()(Args&&... args)
    -> decltype(
         detail::forward_as_tuple(std::forward<Args>(args)...)
       )
  {
    return detail::forward_as_tuple(std::forward<Args>(args)...);
  }
};


template<class Tuple>
__AGENCY_ANNOTATION
auto forward_tail(Tuple&& t)
  -> decltype(
       __tu::tuple_tail_invoke(std::forward<Tuple>(t), forwarder{})
     )
{
  return __tu::tuple_tail_invoke(std::forward<Tuple>(t), forwarder{});
}


struct agency_tuple_maker
{
  template<class... Args>
  __AGENCY_ANNOTATION
  auto operator()(Args&&... args)
    -> decltype(
         agency::detail::make_tuple(std::forward<Args>(args)...)
       )
  {
    return agency::detail::make_tuple(std::forward<Args>(args)...);
  }
};


template<size_t N, class Tuple>
__AGENCY_ANNOTATION
auto tuple_drop(Tuple&& t)
  -> decltype(
       __tu::tuple_drop_invoke<N>(std::forward<Tuple>(t), agency_tuple_maker())
     )
{
  return __tu::tuple_drop_invoke<N>(std::forward<Tuple>(t), agency_tuple_maker());
}


template<class Tuple>
__AGENCY_ANNOTATION
auto tuple_drop_last(Tuple&& t)
  -> decltype(
       agency::detail::tuple_drop<1>(std::forward<Tuple>(t))
     )
{
  return agency::detail::tuple_drop<1>(std::forward<Tuple>(t));
}


template<class Function, class Tuple>
__AGENCY_ANNOTATION
auto tuple_apply(Function f, Tuple&& t)
  -> decltype(
       __tu::tuple_apply(agency::detail::host_device_cast(f), std::forward<Tuple>(t))
     )
{
  return __tu::tuple_apply(agency::detail::host_device_cast(f), std::forward<Tuple>(t));
}


} // end detail
} // end agency

