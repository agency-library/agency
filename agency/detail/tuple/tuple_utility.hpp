#pragma once

#include <agency/detail/config.hpp>

// XXX this #include should be eliminated from this file
#define TUPLE_UTILITY_ANNOTATION __AGENCY_ANNOTATION
#define TUPLE_UTILITY_NAMESPACE __tu
#define TUPLE_UTILITY_NAMESPACE_OPEN_BRACE namespace __tu {
#define TUPLE_UTILITY_NAMESPACE_CLOSE_BRACE }
#include <agency/detail/tuple/tuple_utility_impl.hpp>
#undef TUPLE_UTILITY_ANNOTATION
#undef TUPLE_UTILITY_NAMESPACE
#undef TUPLE_UTILITY_NAMESPACE_OPEN_BRACE
#undef TUPLE_UTILITY_NAMESPACE_CLOSE_BRACE


#include <agency/detail/requires.hpp>
#include <agency/tuple.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/type_list.hpp>
#include <agency/detail/host_device_cast.hpp>
#include <agency/detail/type_traits.hpp>
#include <utility>
#include <type_traits>


namespace agency
{
namespace detail
{


template<class T>
struct is_tuple_like : __tu::is_tuple_like<T> {};


template<class Tuple, class... Types>
struct tuple_rebind : __tu::tuple_rebind<Tuple, Types...> {};


template<class Tuple, class... Types>
using tuple_rebind_t = typename tuple_rebind<Tuple,Types...>::type;


// get_if returns the ith element of an object when that object is a Tuple-like type
// otherwise, it returns its second parameter
template<size_t i, class Tuple, class T,
         __AGENCY_REQUIRES(
           is_tuple_like<typename std::decay<Tuple>::type>::value
         )>
__AGENCY_ANNOTATION
auto get_if(Tuple&& t, T&&)
  -> decltype(agency::get<i>(std::forward<Tuple>(t)))
{
  return agency::get<i>(std::forward<Tuple>(t));
}


template<size_t, class NotATuple, class T,
         __AGENCY_REQUIRES(
           !is_tuple_like<typename std::decay<NotATuple>::type>::value
         )>
__AGENCY_ANNOTATION
T&& get_if(NotATuple&&, T&& otherwise_if_not_tuple)
{
  return std::forward<T>(otherwise_if_not_tuple);
}


// names the ith type of a parameter pack
template<size_t i, class... Types>
struct pack_element
  : std::tuple_element<i,std::tuple<Types...>>
{
};


template<size_t i, class... Types>
using pack_element_t = typename pack_element<i,Types...>::type;


template<class T,
         class = typename std::enable_if<
           is_tuple_like<typename std::decay<T>::type>::value
         >::type>
__AGENCY_ANNOTATION
auto tuple_head_if(T&& t) ->
  decltype(agency::get<0>(std::forward<T>(t)))
{
  return agency::get<0>(std::forward<T>(t));
}


template<class T,
         class = typename std::enable_if<
           !is_tuple_like<typename std::decay<T>::type>::value
         >::type>
__AGENCY_ANNOTATION
T&& tuple_head_if(T&& t)
{
  return std::forward<T>(t);
}


template<class Tuple>
__AGENCY_ANNOTATION
auto tuple_last(Tuple&& t)
  -> decltype(
       agency::get<
         std::tuple_size<
           typename std::decay<Tuple>::type
         >::value - 1
       >(std::forward<Tuple>(t))
     )
{
  constexpr size_t N = std::tuple_size<
    typename std::decay<Tuple>::type
  >::value;

  return agency::get<N - 1>(std::forward<Tuple>(t));
}


template<class T,
         class = typename std::enable_if<
           is_tuple_like<typename std::decay<T>::type>::value
         >::type>
__AGENCY_ANNOTATION
auto tuple_last_if(T&& t) ->
  decltype(agency::detail::tuple_last(std::forward<T>(t)))
{
  return agency::detail::tuple_last(std::forward<T>(t));
}


template<class T,
         class = typename std::enable_if<
           !is_tuple_like<typename std::decay<T>::type>::value
         >::type>
__AGENCY_ANNOTATION
T&& tuple_last_if(T&& t)
{
  return std::forward<T>(t);
}


template<class Function>
struct tuple_all_of_functor
{
  Function f;

  template<class Arg>
  __AGENCY_ANNOTATION
  bool operator()(bool prefix, Arg&& arg) const
  {
    return prefix && f(std::forward<Arg>(arg));
  }
};


template<class Tuple, class Function>
__AGENCY_ANNOTATION
bool tuple_all_of(Tuple&& t, Function f)
{
  return __tu::tuple_reduce(std::forward<Tuple>(t), true, tuple_all_of_functor<Function>{f});
}


template<class Tuple>
using tuple_indices = make_index_sequence<std::tuple_size<Tuple>::value>;


template<class Tuple>
__AGENCY_ANNOTATION
detail::make_index_sequence<
  std::tuple_size<
    typename std::decay<Tuple>::type
  >::value
> 
  make_tuple_indices(Tuple&&)
{
  return detail::make_index_sequence<
    std::tuple_size<
      typename std::decay<Tuple>::type
    >::value
  >();
}


template<class IndexSequence, class Tuple>
struct tuple_elements_impl;

template<size_t... Indices, class Tuple>
struct tuple_elements_impl<index_sequence<Indices...>,Tuple>
{
  using type = type_list<
    typename std::tuple_element<Indices,Tuple>::type...
  >;
};


template<class Tuple>
using tuple_elements = typename tuple_elements_impl<tuple_indices<Tuple>,Tuple>::type;


template<class Tuple,
         class = typename std::enable_if<
           (std::tuple_size<
             typename std::decay<Tuple>::type
           >::value > 1)
         >::type
        >
__AGENCY_ANNOTATION
Tuple&& unwrap_single_element_tuple(Tuple&& t)
{
  return std::forward<Tuple>(t);
}


template<class Tuple,
         class = typename std::enable_if<
           (std::tuple_size<
              typename std::decay<Tuple>::type
           >::value == 1)
         >::type
        >
__AGENCY_ANNOTATION
auto unwrap_single_element_tuple(Tuple&& t)
  -> decltype(
       agency::get<0>(std::forward<Tuple>(t))
     )
{
  return agency::get<0>(std::forward<Tuple>(t));
}


// if the argument is a tuple, it unwraps it if it is a single-element tuple,
// otherwise, it returns the tuple
// if the argument is not a tuple, it returns the argument
template<class Tuple,
         class = typename std::enable_if<
           is_tuple_like<typename std::decay<Tuple>::type>::value
         >::type>
__AGENCY_ANNOTATION
auto unwrap_single_element_tuple_if(Tuple&& t)
  -> decltype(
       detail::unwrap_single_element_tuple(std::forward<Tuple>(t))
     )
{
  return detail::unwrap_single_element_tuple(std::forward<Tuple>(t));
}

template<class T,
         class = typename std::enable_if<
           !is_tuple_like<typename std::decay<T>::type>::value
         >::type>
T&& unwrap_single_element_tuple_if(T&& arg)
{
  return std::forward<T>(arg);
}


template<class Indices, class Tuple>
struct tuple_type_list_impl;

template<size_t... Indices, class Tuple>
struct tuple_type_list_impl<index_sequence<Indices...>, Tuple>
{
  using type = type_list<
    typename std::tuple_element<Indices,Tuple>::type...
  >;
};


template<class T, class Enable = void>
struct tuple_type_list;


template<class Tuple>
struct tuple_type_list<Tuple, typename std::enable_if<is_tuple_like<Tuple>::value>::type>
{
  using type = typename tuple_type_list_impl<
    make_index_sequence<std::tuple_size<Tuple>::value>,
    Tuple
  >::type;
};


template<class>
struct is_empty_tuple;


template<class T>
struct is_empty_tuple_impl_impl;


template<class... Types>
struct is_empty_tuple_impl_impl<type_list<Types...>>
{
  using type = conjunction<
    disjunction<
      std::is_empty<Types>,
      is_empty_tuple<Types>
    >...
  >;
};


template<class T, class Enable = void>
struct is_empty_tuple_impl : std::false_type {};


template<class Tuple>
struct is_empty_tuple_impl<Tuple, typename std::enable_if<is_tuple_like<Tuple>::value>::type>
{
  using type = typename is_empty_tuple_impl_impl<
    typename tuple_type_list<Tuple>::type
  >::type;
};


template<class Tuple>
struct is_empty_tuple : is_empty_tuple_impl<Tuple>::type {};


// a Tuple-like type is rebindable for a list of types if tuple_rebind<Tuple,Types...>::type is detected to exist
// XXX WAR nvbug 1965139
//template<class Tuple, class... Types>
//using is_tuple_like_rebindable = is_detected<tuple_rebind_t, Tuple, Types...>;
template<class Tuple, class... Types>
struct is_tuple_like_rebindable : is_detected<tuple_rebind_t, Tuple, Types...> {};


// some types aren't tuple_rebindable given a list of Types
// in such cases, we default to using the given TupleLike template as the result of the rebind
template<class T, template<class...> class TupleLike, class... Types>
using tuple_rebind_if = lazy_conditional<
  is_tuple_like_rebindable<T,Types...>::value, // if Tuple is rebindable...
  tuple_rebind<T,Types...>,               // then tuple_rebind it
  identity<TupleLike<Types...>>           // otherwise, default to TupleLike<Types...>
>;

template<class T, template<class...> class TupleLike, class... Types>
using tuple_rebind_if_t = typename tuple_rebind_if<T,TupleLike,Types...>::type;


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
         agency::make_tuple(std::forward<Args>(args)...)
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
           is_tuple_like<typename std::decay<T>::type>::value
         >::type>
__AGENCY_ANNOTATION
auto tuple_tail_if(T&& t) ->
  decltype(detail::tuple_tail(std::forward<T>(t)))
{
  return detail::tuple_tail(std::forward<T>(t));
}


template<class T,
         class = typename std::enable_if<
           !is_tuple_like<typename std::decay<T>::type>::value
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
           is_tuple_like<typename std::decay<T>::type>::value
         >::type>
__AGENCY_ANNOTATION
auto tuple_prefix_if(T&& t) ->
  decltype(detail::tuple_prefix(std::forward<T>(t)))
{
  return detail::tuple_prefix(std::forward<T>(t));
}


template<class T,
         class = typename std::enable_if<
           !is_tuple_like<typename std::decay<T>::type>::value
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
           is_tuple_like<typename std::decay<T>::type>::value
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
           !is_tuple_like<typename std::decay<T>::type>::value
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


template<class T, class Tuple>
__AGENCY_ANNOTATION
T make_from_tail(Tuple&& t)
{
  return __tu::tuple_tail_invoke(std::forward<Tuple>(t), detail::maker<T>());
}


} // end detail
} // end agency

