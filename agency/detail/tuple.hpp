#pragma once

#include <agency/detail/config.hpp>

#define __TUPLE_ANNOTATION __AGENCY_ANNOTATION

#define __TUPLE_NAMESPACE __tu

#include <agency/detail/tuple_impl.hpp>
#include <agency/detail/tuple_utility.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/type_list.hpp>
#include <agency/detail/host_device_cast.hpp>
#include <utility>
#include <type_traits>

namespace __tu
{


// add an output operator for tuple
template<class... Types>
std::ostream& operator<<(std::ostream& os, const tuple<Types...>& t)
{
  os << "{";
  __tu::tuple_print(t, os);
  os << "}";
  return os;
}


} // end namespace __tu


namespace agency
{
namespace detail
{


template<class... Types>
using tuple = __tu::tuple<Types...>;
using __tu::swap;
using __tu::make_tuple;
using __tu::tie;
using __tu::forward_as_tuple;


using ignore_t = decltype(__tu::ignore);
constexpr ignore_t ignore{};


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


struct tuple_mover
{
  template<class... Args>
  __AGENCY_ANNOTATION
  auto operator()(Args&&... args)
    -> decltype(
         detail::make_tuple(std::move(args)...)
       )
  {
    return detail::make_tuple(std::move(args)...);
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
         agency::detail::make_tuple(std::forward<Args>(args)...)
       )
  {
    return agency::detail::make_tuple(std::forward<Args>(args)...);
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


template<typename Function, typename Tuple, typename... Tuples>
__AGENCY_ANNOTATION
auto tuple_map(Function f, Tuple&& t, Tuples&&... ts)
  -> decltype(
       __tu::tuple_map_with_make(f, agency_tuple_maker(), std::forward<Tuple>(t), std::forward<Tuples>(ts)...)
     )
{
  return __tu::tuple_map_with_make(f, agency_tuple_maker(), std::forward<Tuple>(t), std::forward<Tuples>(ts)...);
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


template<size_t N, class Tuple>
__AGENCY_ANNOTATION
auto tuple_take_view(Tuple&& t)
  -> decltype(
       __tu::tuple_take_invoke<N>(std::forward<Tuple>(t), forwarder())
     )
{
  return __tu::tuple_take_invoke<N>(std::forward<Tuple>(t), forwarder());
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


template<class Function, class Tuple>
__AGENCY_ANNOTATION
auto tuple_apply(Function f, Tuple&& t)
  -> decltype(
       __tu::tuple_apply(agency::detail::host_device_cast(f), std::forward<Tuple>(t))
     )
{
  return __tu::tuple_apply(agency::detail::host_device_cast(f), std::forward<Tuple>(t));
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


template<size_t... Indices, class Tuple>
__AGENCY_ANNOTATION
auto tuple_gather(Tuple&& t)
  -> decltype(
       __tu::tuple_gather_invoke<Indices...>(std::forward<Tuple>(t), agency_tuple_maker())
     )
{
  return __tu::tuple_gather_invoke<Indices...>(std::forward<Tuple>(t), agency_tuple_maker());
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


template<class Tuple,
         class = typename std::enable_if<
           (std::tuple_size<
             typename std::decay<Tuple>::type
           >::value > 1)
         >::type
        >
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
auto unwrap_single_element_tuple(Tuple&& t)
  -> decltype(
       detail::get<0>(std::forward<Tuple>(t))
     )
{
  return detail::get<0>(std::forward<Tuple>(t));
}


template<class TupleReference, class IndexSequence>
struct decay_tuple_impl;


template<class TupleReference, size_t... Indices>
struct decay_tuple_impl<TupleReference, index_sequence<Indices...>>
{
  using tuple_type = typename std::decay<TupleReference>::type;

  using type = detail::tuple<
    typename std::decay<
      typename std::tuple_element<
        Indices,
        tuple_type
      >::type
    >::type...
  >;
};


template<class TupleReference>
struct decay_tuple : decay_tuple_impl<TupleReference, tuple_indices<typename std::decay<TupleReference>::type>> {};

template<class TupleReference>
using decay_tuple_t = typename decay_tuple<TupleReference>::type;


template<class TypeList>
struct homogeneous_tuple_impl;

template<class... Types>
struct homogeneous_tuple_impl<type_list<Types...>>
{
  using type = tuple<Types...>;
};

template<class T, size_t size>
using homogeneous_tuple = typename homogeneous_tuple_impl<type_list_repeat<size,T>>:: type;


template<size_t size, class T>
__AGENCY_ANNOTATION
homogeneous_tuple<T,size> make_homogeneous_tuple(const T& val)
{
  return detail::tuple_repeat<size>(val);
}



} // end detail
} // end agency

