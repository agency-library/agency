#pragma once

#include <agency/detail/config.hpp>
#include <agency/tuple/detail/tuple_utility.hpp>
#include <tuple>
#include <utility>
#include <type_traits>


// XXX move this content up into agency/tuple.hpp
//     there's not really a need to reserve a separate directory for tuple


namespace agency
{


template<class... Types>
using tuple = __tu::tuple<Types...>;


namespace detail
{


struct ignore_t
{
  template<class T>
  __AGENCY_ANNOTATION
  const ignore_t operator=(T&&) const
  {
    return *this;
  }
};


} // end detail


constexpr detail::ignore_t ignore{};


template<class... Types>
__AGENCY_ANNOTATION
void swap(tuple<Types...>& a, tuple<Types...>& b)
{
  a.swap(b);
}


template<class... Types>
__AGENCY_ANNOTATION
tuple<typename std::decay<Types>::type...> make_tuple(Types&&... args)
{
  return tuple<typename std::decay<Types>::type...>(std::forward<Types>(args)...);
}


template<class... Types>
__AGENCY_ANNOTATION
tuple<Types&...> tie(Types&... args)
{
  return tuple<Types&...>(args...);
}


template<class... Args>
__AGENCY_ANNOTATION
tuple<Args&&...> forward_as_tuple(Args&&... args)
{
  return tuple<Args&&...>(std::forward<Args>(args)...);
}


namespace detail
{


template<class IndexSequence, class... Tuples>
struct tuple_cat_result_impl_impl;


template<size_t... I, class... Tuples>
struct tuple_cat_result_impl_impl<index_sequence<I...>, Tuples...>
{
  using type = tuple<typename __tu::__tuple_cat_element<I, Tuples...>::type...>;
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


template<class T>
struct maker
{
  template<class... Args>
  __AGENCY_ANNOTATION
  T operator()(Args&&... args)
  {
    return T{std::forward<Args>(args)...};
  }
};


// XXX this should be moved into some file underneath agency/detail/tuple
template<class T, class Tuple>
__AGENCY_ANNOTATION
T make_from_tail(Tuple&& t)
{
  return __tu::tuple_tail_invoke(std::forward<Tuple>(t), detail::maker<T>());
}


} // end detail


// XXX this doesn't forward tuple elements which are reference types correctly
//     because make_tuple() doesn't do that
template<class... Tuples>
__AGENCY_ANNOTATION
detail::tuple_cat_result<Tuples...> tuple_cat(Tuples&&... tuples)
{
  return __tu::tuple_cat_apply(detail::maker<detail::tuple_cat_result<Tuples...>>{}, std::forward<Tuples>(tuples)...);
}


// an output operator for tuple
template<class... Types>
std::ostream& operator<<(std::ostream& os, const tuple<Types...>& t)
{
  os << "{";
  __tu::tuple_print(t, os);
  os << "}";
  return os;
}


// XXX move this stuff into detail/tuple_utility.hpp
//     when that is possible


namespace detail
{


struct forwarder
{
  template<class... Args>
  __AGENCY_ANNOTATION
  auto operator()(Args&&... args)
    -> decltype(
         agency::forward_as_tuple(std::forward<Args>(args)...)
       )
  {
    return agency::forward_as_tuple(std::forward<Args>(args)...);
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


struct tuple_mover
{
  template<class... Args>
  __AGENCY_ANNOTATION
  auto operator()(Args&&... args)
    -> decltype(
         agency::make_tuple(std::move(args)...)
       )
  {
    return agency::make_tuple(std::move(args)...);
  }
};


template<class Tuple>
__AGENCY_ANNOTATION
auto move_tail(Tuple&& t)
  -> decltype(
       __tu::tuple_tail_invoke(std::move(t), tuple_mover{})
     )
{
  return __tu::tuple_tail_invoke(std::move(t), tuple_mover{});
}


struct agency_tuple_maker
{
  template<class... Args>
  __AGENCY_ANNOTATION
  auto operator()(Args&&... args)
    -> decltype(
         __tu::make_tuple(std::forward<Args>(args)...)
       )
  {
    return agency::make_tuple(std::forward<Args>(args)...);
  }
};


template<class Tuple>
__AGENCY_ANNOTATION
auto tuple_tail(Tuple&& t)
  -> decltype(
       __tu::tuple_tail_invoke(std::forward<Tuple>(t), agency_tuple_maker{})
     )
{
  return __tu::tuple_tail_invoke(std::forward<Tuple>(t), agency_tuple_maker{});
}


template<class T,
         class = typename std::enable_if<
           is_tuple<typename std::decay<T>::type>::value
         >::type>
__AGENCY_ANNOTATION
auto tuple_tail_if(T&& t) ->
  decltype(detail::tuple_tail(std::forward<T>(t)))
{
  return detail::tuple_tail(std::forward<T>(t));
}


template<class T,
         class = typename std::enable_if<
           !is_tuple<typename std::decay<T>::type>::value
         >::type>
__AGENCY_ANNOTATION
agency::tuple<> tuple_tail_if(T&&)
{
  return agency::tuple<>();
}


template<class Tuple>
__AGENCY_ANNOTATION
auto tuple_prefix(Tuple&& t)
  -> decltype(
       __tu::tuple_prefix_invoke(std::forward<Tuple>(t), agency_tuple_maker{})
     )
{
  return __tu::tuple_prefix_invoke(std::forward<Tuple>(t), agency_tuple_maker{});
}


template<class T,
         class = typename std::enable_if<
           is_tuple<typename std::decay<T>::type>::value
         >::type>
__AGENCY_ANNOTATION
auto tuple_prefix_if(T&& t) ->
  decltype(detail::tuple_prefix(std::forward<T>(t)))
{
  return detail::tuple_prefix(std::forward<T>(t));
}


template<class T,
         class = typename std::enable_if<
           !is_tuple<typename std::decay<T>::type>::value
         >::type>
__AGENCY_ANNOTATION
agency::tuple<> tuple_prefix_if(T&&)
{
  return agency::tuple<>();
}


template<size_t N, class Tuple>
__AGENCY_ANNOTATION
auto tuple_take(Tuple&& t)
  -> decltype(
       __tu::tuple_take_invoke<N>(std::forward<Tuple>(t), agency_tuple_maker())
     )
{
  return __tu::tuple_take_invoke<N>(std::forward<Tuple>(t), agency_tuple_maker());
}


template<size_t n,
         class T,
         class = typename std::enable_if<
           is_tuple<typename std::decay<T>::type>::value
         >::type,
         class = typename std::enable_if<
           (n <= std::tuple_size<typename std::decay<T>::type>::value)
         >::type>
__AGENCY_ANNOTATION
auto tuple_take_if(T&& t) ->
  decltype(detail::tuple_take<n>(std::forward<T>(t)))
{
  return detail::tuple_take<n>(std::forward<T>(t));
}


template<size_t N, class Tuple>
__AGENCY_ANNOTATION
auto tuple_take_view(Tuple&& t)
  -> decltype(
       __tu::tuple_take_invoke<N>(std::forward<Tuple>(t), forwarder())
     )
{
  return __tu::tuple_take_invoke<N>(std::forward<Tuple>(t), forwarder());
}


template<size_t n,
         class T,
         class = typename std::enable_if<
           !is_tuple<typename std::decay<T>::type>::value
         >::type,
         class = typename std::enable_if<
           n == 1
         >::type>
__AGENCY_ANNOTATION
typename std::decay<T>::type tuple_take_if(T&& value)
{
  return std::forward<T>(value);
}


template<size_t N, class Tuple>
__AGENCY_ANNOTATION
auto tuple_drop(Tuple&& t)
  -> decltype(
       __tu::tuple_drop_invoke<N>(std::forward<Tuple>(t), agency_tuple_maker())
     )
{
  return __tu::tuple_drop_invoke<N>(std::forward<Tuple>(t), agency_tuple_maker());
}


template<size_t N, class Tuple>
__AGENCY_ANNOTATION
auto tuple_drop_view(Tuple&& t)
  -> decltype(
       __tu::tuple_drop_invoke<N>(std::forward<Tuple>(t), forwarder())
     )
{
  return __tu::tuple_drop_invoke<N>(std::forward<Tuple>(t), forwarder());
}


template<size_t N, class Tuple>
__AGENCY_ANNOTATION
auto tuple_drop_back(Tuple&& t)
  -> decltype(
       __tu::tuple_drop_back_invoke<N>(std::forward<Tuple>(t), agency_tuple_maker())
     )
{
  return __tu::tuple_drop_back_invoke<N>(std::forward<Tuple>(t), agency_tuple_maker());
}


template<class Tuple>
__AGENCY_ANNOTATION
auto tuple_drop_last(Tuple&& t)
  -> decltype(
       agency::detail::tuple_drop_back<1>(std::forward<Tuple>(t))
     )
{
  return agency::detail::tuple_drop_back<1>(std::forward<Tuple>(t));
}


template<size_t N, class T>
__AGENCY_ANNOTATION
auto tuple_repeat(const T& x)
  -> decltype(
       __tu::tuple_repeat_invoke<N>(x, agency_tuple_maker())
     )
{
  return __tu::tuple_repeat_invoke<N>(x, agency_tuple_maker());
}


template<template<class T> class MetaFunction, class Tuple>
__AGENCY_ANNOTATION
auto tuple_filter(Tuple&& t)
  -> decltype(
       __tu::tuple_filter_invoke<MetaFunction>(std::forward<Tuple>(t), agency_tuple_maker())
     )
{
  return __tu::tuple_filter_invoke<MetaFunction>(std::forward<Tuple>(t), agency_tuple_maker());
}


template<template<class T> class MetaFunction, class Tuple>
__AGENCY_ANNOTATION
auto tuple_filter_view(Tuple&& t)
  -> decltype(
        __tu::tuple_filter_invoke<MetaFunction>(std::forward<Tuple>(t), forwarder())
     )
{
  return __tu::tuple_filter_invoke<MetaFunction>(std::forward<Tuple>(t), forwarder());
}


template<typename Function, typename Tuple, typename... Tuples>
__AGENCY_ANNOTATION
auto tuple_map(Function f, Tuple&& t, Tuples&&... ts)
  -> decltype(
       __tu::tuple_map_with_make(f, agency_tuple_maker(), std::forward<Tuple>(t), std::forward<Tuples>(ts)...)
     )
{
  return __tu::tuple_map_with_make(f, agency_tuple_maker(), std::forward<Tuple>(t), std::forward<Tuples>(ts)...);
}


template<size_t... Indices, class Tuple>
__AGENCY_ANNOTATION
auto tuple_gather(Tuple&& t)
  -> decltype(
       __tu::tuple_gather_invoke<Indices...>(std::forward<Tuple>(t), agency_tuple_maker())
     )
{
  return __tu::tuple_gather_invoke<Indices...>(std::forward<Tuple>(t), agency_tuple_maker());
}


template<class Tuple, class T>
__AGENCY_ANNOTATION
auto tuple_append(Tuple&& t, T&& val)
  -> decltype(
       __tu::tuple_append_invoke(std::forward<Tuple>(t), std::forward<T>(val), agency_tuple_maker())
     )
{
  return __tu::tuple_append_invoke(std::forward<Tuple>(t), std::forward<T>(val), agency_tuple_maker());
}


template<class Tuple, class T>
__AGENCY_ANNOTATION
auto tuple_prepend(Tuple&& t, T&& val)
  -> decltype(
       __tu::tuple_prepend_invoke(std::forward<Tuple>(t), std::forward<T>(val), agency_tuple_maker())
     )
{
  return __tu::tuple_prepend_invoke(std::forward<Tuple>(t), std::forward<T>(val), agency_tuple_maker());
}


template<class Tuple, class T>
struct tuple_prepend_result
{
  using type = decltype(
    detail::tuple_prepend(
      std::declval<Tuple>(),
      std::declval<T>()
    )
  );
};

template<class Tuple, class T>
using tuple_prepend_result_t = typename tuple_prepend_result<Tuple,T>::type;


} // end detail
} // end agency

