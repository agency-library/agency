// Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <tuple>
#include <utility>
#include <type_traits>
#include <iostream>

// allow the user to define an annotation to apply to these functions
// by default, it attempts to be constexpr
#ifndef TUPLE_UTILITY_ANNOTATION
#  if __cplusplus <= 201103L
#    define TUPLE_UTILITY_ANNOTATION
#  else
#    define TUPLE_UTILITY_ANNOTATION constexpr
#  endif
#  define TUPLE_UTILITY_ANNOTATION_NEEDS_UNDEF
#endif

// allow the user to define a namespace for these functions
#ifdef TUPLE_UTILITY_NAMESPACE
namespace TUPLE_UTILITY_NAMESPACE
{
#endif // TUPLE_UTILITY_NAMESPACE


template<class T>
using __decay_t = typename std::decay<T>::type;


// define index sequence in case it is missing
template<size_t... I> struct __index_sequence {};

template<size_t Start, typename Indices, size_t End>
struct __make_index_sequence_impl;

template<size_t Start, size_t... Indices, size_t End>
struct __make_index_sequence_impl<
  Start,
  __index_sequence<Indices...>, 
  End
>
{
  typedef typename __make_index_sequence_impl<
    Start + 1,
    __index_sequence<Indices..., Start>,
    End
  >::type type;
};

template<size_t End, size_t... Indices>
struct __make_index_sequence_impl<End, __index_sequence<Indices...>, End>
{
  typedef __index_sequence<Indices...> type;
};

template<size_t N>
using __make_index_sequence = typename __make_index_sequence_impl<0, __index_sequence<>, N>::type;


template<class T>
struct tuple_traits
{
  using tuple_type = T;

  static const size_t size = std::tuple_size<tuple_type>::value; 

  template<size_t i>
  using element_type = typename std::tuple_element<i,tuple_type>::type;

  template<size_t i>
  TUPLE_UTILITY_ANNOTATION
  static element_type<i>& get(tuple_type& t)
  {
    return std::get<i>(t);
  }

  template<size_t i>
  TUPLE_UTILITY_ANNOTATION
  static const element_type<i>& get(const tuple_type& t)
  {
    return std::get<i>(t);
  }

  template<size_t i>
  TUPLE_UTILITY_ANNOTATION
  static element_type<i>&& get(tuple_type&& t)
  {
    return std::get<i>(std::move(t));
  }
};


template<size_t i, class Tuple>
TUPLE_UTILITY_ANNOTATION
auto __get(Tuple&& t)
  -> decltype(
       tuple_traits<__decay_t<Tuple>>::template get<i>(std::forward<Tuple>(t))
     )
{
  return tuple_traits<__decay_t<Tuple>>::template get<i>(std::forward<Tuple>(t));
}


template<class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_head(Tuple&& t)
  -> decltype(
       __get<0>(std::forward<Tuple>(t))
     )
{
  return __get<0>(std::forward<Tuple>(t));
}


template<class Tuple, class Function, size_t... I>
TUPLE_UTILITY_ANNOTATION
auto __tuple_tail_invoke_impl(Tuple&& t, Function f, __index_sequence<I...>)
  -> decltype(
       f(
         __get<I+1>(std::forward<Tuple>(t))...
       )
     )
{
  return f(__get<I+1>(std::forward<Tuple>(t))...);
}


template<class Tuple, class Function>
TUPLE_UTILITY_ANNOTATION
auto tuple_tail_invoke(Tuple&& t, Function f)
  -> decltype(
       __tuple_tail_invoke_impl(
         std::forward<Tuple>(t),
         f,
         __make_index_sequence<
           std::tuple_size<
             typename std::decay<Tuple>::type
           >::value - 1
         >()
       )
     )
{
  return __tuple_tail_invoke_impl(
    std::forward<Tuple>(t),
    f,
    __make_index_sequence<
      std::tuple_size<
        typename std::decay<Tuple>::type
      >::value - 1
    >()
  );
}


struct __tuple_forwarder
{
  template<class... Args>
  TUPLE_UTILITY_ANNOTATION
  auto operator()(Args&&... args)
    -> decltype(
         std::forward_as_tuple(args...)
       )
  {
    return std::forward_as_tuple(args...);
  }
};


// forward_tuple_tail() returns t's tail as a tuple of references
template<class Tuple>
TUPLE_UTILITY_ANNOTATION
auto forward_tuple_tail(typename std::remove_reference<Tuple>::type& t)
  -> decltype(
       tuple_tail_invoke(std::forward<Tuple>(t), __tuple_forwarder{})
     )
{
  return tuple_tail_invoke(std::forward<Tuple>(t), __tuple_forwarder{});
}


template<class... Args>
TUPLE_UTILITY_ANNOTATION
auto __tuple_tail_impl_impl(Args&&... args)
  -> decltype(
       std::make_tuple(args...)
     )
{
  return std::make_tuple(args...);
}

template<class Tuple, size_t... I>
TUPLE_UTILITY_ANNOTATION
auto __tuple_tail_impl(Tuple&& t, __index_sequence<I...>)
  -> decltype(
       __tuple_tail_impl_impl(__get<I+1>(std::forward<Tuple>(t))...)
     )
{
  return __tuple_tail_impl_impl(__get<I+1>(std::forward<Tuple>(t))...);
}


// tuple_tail() returns t's tail as a tuple of values
// i.e., it makes a copy of t's tail
template<class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_tail(Tuple&& t)
  -> decltype(
       __tuple_tail_impl(
         std::forward<Tuple>(t),
         __make_index_sequence<
           std::tuple_size<
             typename std::decay<Tuple>::type
           >::value - 1
         >()
       )
     )
{
  using indices = __make_index_sequence<
    std::tuple_size<
      typename std::decay<Tuple>::type
    >::value - 1
  >;
  return __tuple_tail_impl(std::forward<Tuple>(t), indices());
}


template<class Tuple, class Function, size_t... I>
TUPLE_UTILITY_ANNOTATION
auto __tuple_take_invoke_impl(Tuple&& t, Function f, __index_sequence<I...>)
  -> decltype(
       f(
         __get<I>(std::forward<Tuple>(t))...
       )
     )
{
  return f(__get<I>(std::forward<Tuple>(t))...);
}


template<size_t N, class Tuple, class Function>
TUPLE_UTILITY_ANNOTATION
auto tuple_take_invoke(Tuple&& t, Function f)
  -> decltype(
       __tuple_take_invoke_impl(std::forward<Tuple>(t), f, __make_index_sequence<N>())
     )
{
  return __tuple_take_invoke_impl(std::forward<Tuple>(t), f, __make_index_sequence<N>());
}


struct __std_tuple_maker
{
  template<class... T>
  TUPLE_UTILITY_ANNOTATION
  auto operator()(T&&... args)
    -> decltype(
         std::make_tuple(std::forward<T>(args)...)
       )
  {
    return std::make_tuple(std::forward<T>(args)...);
  }
};


template<size_t N, class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_take(Tuple&& t)
  -> decltype(
       tuple_take_invoke<N>(std::forward<Tuple>(t), __std_tuple_maker())
     )
{
  return tuple_take_invoke<N>(std::forward<Tuple>(t), __std_tuple_maker());
}


template<size_t N, class Tuple, class Function>
TUPLE_UTILITY_ANNOTATION
auto tuple_drop_invoke(Tuple&& t, Function f)
  -> decltype(
       tuple_take_invoke<std::tuple_size<typename std::decay<Tuple>::type>::value - N>(std::forward<Tuple>(t), f)
     )
{
  return tuple_take_invoke<std::tuple_size<typename std::decay<Tuple>::type>::value - N>(std::forward<Tuple>(t), f);
}


template<size_t N, class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_drop(Tuple&& t)
  -> decltype(
       tuple_drop_invoke<N>(std::forward<Tuple>(t), __std_tuple_maker())
     )
{
  return tuple_drop_invoke<N>(std::forward<Tuple>(t), __std_tuple_maker());
}


template<class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_drop_last(Tuple&& t)
  -> decltype(
       tuple_drop<1>(std::forward<Tuple>(t))
     )
{
  return tuple_drop<1>(std::forward<Tuple>(t));
}


template<class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_last(Tuple&& t)
  -> decltype(
       __get<
         std::tuple_size<
           typename std::decay<Tuple>::type
         >::value - 1
       >(std::forward<Tuple>(t))
     )
{
  const size_t i = std::tuple_size<typename std::decay<Tuple>::type>::value - 1;
  return __get<i>(std::forward<Tuple>(t));
}


template<class Tuple, class T, class Function, size_t... I>
TUPLE_UTILITY_ANNOTATION
auto __tuple_append_invoke_impl(Tuple&& t, T&& x, Function f, __index_sequence<I...>)
  -> decltype(
       f(__get<I>(std::forward<Tuple>(t))..., std::forward<T>(x))
     )
{
  return f(__get<I>(std::forward<Tuple>(t))..., std::forward<T>(x));
}


template<class Tuple, class T, class Function>
TUPLE_UTILITY_ANNOTATION
auto tuple_append_invoke(Tuple&& t, T&& x, Function f)
  -> decltype(
       __tuple_append_invoke_impl(
         std::forward<Tuple>(t),
         std::forward<T>(x),
         f,
         __make_index_sequence<
           std::tuple_size<
             typename std::decay<Tuple>::type
           >::value
         >()
       )
     )
{
  using indices = __make_index_sequence<
    std::tuple_size<typename std::decay<Tuple>::type>::value
  >;
  return __tuple_append_invoke_impl(std::forward<Tuple>(t), std::forward<T>(x), f, indices());
}


template<class Tuple, class T>
TUPLE_UTILITY_ANNOTATION
auto tuple_append(Tuple&& t, T&& x)
  -> decltype(
       tuple_append_invoke(std::forward<Tuple>(t), std::forward<T>(x), __std_tuple_maker())
     )
{
  return tuple_append_invoke(std::forward<Tuple>(t), std::forward<T>(x), __std_tuple_maker());
}



template<size_t I, typename Function, typename... Tuples>
TUPLE_UTILITY_ANNOTATION
auto __tuple_map_invoke(Function f, Tuples&&... ts)
  -> decltype(
       f(__get<I>(std::forward<Tuples>(ts))...)
     )
{
  return f(__get<I>(std::forward<Tuples>(ts))...);
}


template<size_t... I, typename Function1, typename Function2, typename... Tuples>
TUPLE_UTILITY_ANNOTATION
auto __tuple_map_with_make_impl(__index_sequence<I...>, Function1 f, Function2 make, Tuples&&... ts)
  -> decltype(
       make(
         __tuple_map_invoke<I>(f, std::forward<Tuples>(ts)...)...
       )
     )
{
  return make(
    __tuple_map_invoke<I>(f, std::forward<Tuples>(ts)...)...
  );
}


template<typename Function1, typename Function2, typename Tuple, typename... Tuples>
TUPLE_UTILITY_ANNOTATION
auto tuple_map_with_make(Function1 f, Function2 make, Tuple&& t, Tuples&&... ts)
  -> decltype(
       __tuple_map_with_make_impl(
         __make_index_sequence<
           std::tuple_size<__decay_t<Tuple>>::value
         >(),
         f,
         make,
         std::forward<Tuple>(t),
         std::forward<Tuples>(ts)...
       )
     )
{
  return __tuple_map_with_make_impl(
    __make_index_sequence<
      std::tuple_size<__decay_t<Tuple>>::value
    >(),
    f,
    make,
    std::forward<Tuple>(t),
    std::forward<Tuples>(ts)...
  );
}


template<typename Function, typename Tuple, typename... Tuples>
TUPLE_UTILITY_ANNOTATION
auto tuple_map(Function f, Tuple&& t, Tuples&&... ts)
  -> decltype(
       tuple_map_with_make(f, __std_tuple_maker(), std::forward<Tuple>(t), std::forward<Tuples>(ts)...)
     )
{
  return tuple_map_with_make(f, __std_tuple_maker(), std::forward<Tuple>(t), std::forward<Tuples>(ts)...);
}


template<class T, class Tuple, size_t... I>
TUPLE_UTILITY_ANNOTATION
T make_from_tuple_impl(const Tuple& t, __index_sequence<I...>)
{
  return T{__get<I>(t)...};
}


template<class T, class Tuple>
TUPLE_UTILITY_ANNOTATION
T make_from_tuple(const Tuple& t)
{
  return make_from_tuple_impl<T>(t, __make_index_sequence<std::tuple_size<Tuple>::value>());
}


template<size_t i, class Tuple, class T, class Function>
TUPLE_UTILITY_ANNOTATION
typename std::enable_if<
  (i >= std::tuple_size<
     typename std::decay<Tuple>::type
   >::value),
   T
>::type
  __tuple_reduce_impl(Tuple&&, T init, Function)
{
  return init;
}



template<size_t i, class Tuple, class T, class Function>
TUPLE_UTILITY_ANNOTATION
typename std::enable_if<
  (i < std::tuple_size<
     typename std::decay<Tuple>::type
   >::value),
   T
>::type
  __tuple_reduce_impl(Tuple&& t, T init, Function f)
{
  return f(
    __get<i>(std::forward<Tuple>(t)),
    __tuple_reduce_impl<i+1>(std::forward<Tuple>(t), init, f)
  );
}


template<class Tuple, class T, class Function>
TUPLE_UTILITY_ANNOTATION
T tuple_reduce(Tuple&& t, T init, Function f)
{
  return __tuple_reduce_impl<0>(std::forward<Tuple>(t), init, f);
}


template<size_t I, size_t N>
struct __tuple_for_each_impl
{
  template<class Function, class Tuple1, class... Tuples>
  TUPLE_UTILITY_ANNOTATION
  static void for_each(Function f, Tuple1&& t1, Tuples&&... ts)
  {
    f(__get<I>(std::forward<Tuple1>(t1)), __get<I>(std::forward<Tuples>(ts))...);

    return __tuple_for_each_impl<I+1,N>::for_each(f, std::forward<Tuple1>(t1), std::forward<Tuples>(ts)...);
  }
};


template<size_t I>
struct __tuple_for_each_impl<I,I>
{
  template<class Function, class Tuple1, class... Tuples>
  TUPLE_UTILITY_ANNOTATION
  static void for_each(Function f, Tuple1&&, Tuples&&...)
  {
    return;
  }
};


template<class Function, class Tuple1, class... Tuples>
TUPLE_UTILITY_ANNOTATION
void tuple_for_each(Function f, Tuple1&& t1, Tuples&&... ts)
{
  return __tuple_for_each_impl<0,std::tuple_size<__decay_t<Tuple1>>::value>::for_each(f, std::forward<Tuple1>(t1), std::forward<Tuples>(ts)...);
}


template<class Tuple, class T>
typename std::enable_if<
  std::tuple_size<Tuple>::value == 0
>::type
tuple_print(const Tuple& t, std::ostream& os, const T&)
{
}


template<class Tuple, class T>
typename std::enable_if<
  std::tuple_size<Tuple>::value == 1
>::type
tuple_print(const Tuple& t, std::ostream& os, const T&)
{
  os << __get<0>(t);
}


template<class Tuple, class T>
typename std::enable_if<
  (std::tuple_size<Tuple>::value > 1)
>::type
  tuple_print(const Tuple& t, std::ostream& os, const T& delimiter)
{
  os << tuple_head(t) << delimiter;

  tuple_print(forward_tuple_tail<const Tuple&>(t), os, delimiter);
}

template<class Tuple>
void tuple_print(const Tuple& t, std::ostream& os = std::cout)
{
  tuple_print(t, os, ", ");
}


template<class Tuple1, class Tuple2>
TUPLE_UTILITY_ANNOTATION
typename std::enable_if<
  std::tuple_size<Tuple1>::value != std::tuple_size<Tuple1>::value,
  bool
>::type
  tuple_equal(const Tuple1&, const Tuple2&)
{
  return false;
}


template<class Tuple1, class Tuple2>
TUPLE_UTILITY_ANNOTATION
typename std::enable_if<
  std::tuple_size<Tuple1>::value == 0,
  bool
>::type
  tuple_equal(const Tuple1&, const Tuple2&)
{
  return true;
}


template<class Tuple1, class Tuple2>
TUPLE_UTILITY_ANNOTATION
typename std::enable_if<
  (std::tuple_size<Tuple1>::value > 0),
  bool
>::type
  tuple_equal(const Tuple1& t1, const Tuple2& t2)
{
  return (tuple_head(t1) != tuple_head(t2)) ? false :
         tuple_equal(forward_tuple_tail<const Tuple1>(t1), forward_tuple_tail<const Tuple2>(t2));
}


template<class Tuple1, class Tuple2>
TUPLE_UTILITY_ANNOTATION
typename std::enable_if<
  std::tuple_size<Tuple2>::value == 0,
  bool
>::type
  tuple_lexicographical_compare(const Tuple1&, const Tuple2&)
{
  return false;
}


template<class Tuple1, class Tuple2>
TUPLE_UTILITY_ANNOTATION
typename std::enable_if<
  (std::tuple_size<Tuple1>::value == 0 && std::tuple_size<Tuple2>::value > 0),
  bool
>::type
  tuple_lexicographical_compare(const Tuple1& t1, const Tuple2& t2)
{
  return true;
}


template<class Tuple1, class Tuple2>
TUPLE_UTILITY_ANNOTATION
typename std::enable_if<
  (std::tuple_size<Tuple1>::value > 0 && std::tuple_size<Tuple2>::value > 0),
  bool
>::type
  tuple_lexicographical_compare(const Tuple1& t1, const Tuple2& t2)
{
  return (tuple_head(t1) < tuple_head(t2)) ? true :
         (tuple_head(t2) < tuple_head(t1)) ? false :
         tuple_lexicographical_compare(forward_tuple_tail<const Tuple1>(t1), forward_tuple_tail<const Tuple2>(t2));
}


template<bool b, class True, class False>
struct __lazy_conditional
{
  using type = typename True::type;
};


template<class True, class False>
struct __lazy_conditional<false, True, False>
{
  using type = typename False::type;
};


template<size_t I, class Tuple1, class... Tuples>
struct __tuple_cat_get_result
{
  using tuple1_type = typename std::decay<Tuple1>::type;
  static const size_t size1 = std::tuple_size<typename std::decay<Tuple1>::type>::value;

  using type = typename __lazy_conditional<
    (I < size1),
    std::tuple_element<I,tuple1_type>,
    __tuple_cat_get_result<I - size1, Tuples...>
  >::type;
};


template<size_t I, class Tuple1>
struct __tuple_cat_get_result<I,Tuple1>
  : std::tuple_element<I, typename std::decay<Tuple1>::type>
{};


template<size_t I, class Tuple1, class... Tuples>
TUPLE_UTILITY_ANNOTATION
typename __tuple_cat_get_result<I,Tuple1,Tuples...>::type
  __tuple_cat_get(Tuple1&& t, Tuples&&... ts);


template<size_t I, class Tuple1, class... Tuples>
TUPLE_UTILITY_ANNOTATION
typename __tuple_cat_get_result<I,Tuple1,Tuples...>::type
  __tuple_cat_get_impl(std::false_type, Tuple1&& t, Tuples&&...)
{
  return __get<I>(std::forward<Tuple1>(t));
}


template<size_t I, class Tuple1, class... Tuples>
TUPLE_UTILITY_ANNOTATION
typename __tuple_cat_get_result<I,Tuple1,Tuples...>::type
  __tuple_cat_get_impl(std::true_type, Tuple1&&, Tuples&&... ts)
{
  const size_t J = I - std::tuple_size<typename std::decay<Tuple1>::type>::value;
  return __tuple_cat_get<J>(std::forward<Tuples>(ts)...);
}


template<size_t I, class Tuple1, class... Tuples>
TUPLE_UTILITY_ANNOTATION
typename __tuple_cat_get_result<I,Tuple1,Tuples...>::type
  __tuple_cat_get(Tuple1&& t, Tuples&&... ts)
{
  auto recurse = typename std::conditional<
    I < std::tuple_size<typename std::decay<Tuple1>::type>::value,
    std::false_type,
    std::true_type
  >::type();

  return __tuple_cat_get_impl<I>(recurse, std::forward<Tuple1>(t), std::forward<Tuples>(ts)...);
}


template<size_t... I, class Function, class... Tuples>
TUPLE_UTILITY_ANNOTATION
auto __tuple_cat_apply_impl(__index_sequence<I...>, Function f, Tuples&&... ts)
  -> decltype(
       f(__tuple_cat_get<I>(std::forward<Tuples>(ts)...)...)
     )
{
  return f(__tuple_cat_get<I>(std::forward<Tuples>(ts)...)...);
}


template<size_t Size, size_t... Sizes>
struct __sum
  : std::integral_constant<
      size_t,
      Size + __sum<Sizes...>::value
    >
{};


template<size_t Size> struct __sum<Size> : std::integral_constant<size_t, Size> {};


template<class Function, class... Tuples>
TUPLE_UTILITY_ANNOTATION
auto tuple_cat_apply(Function f, Tuples&&... ts)
  -> decltype(
       __tuple_cat_apply_impl(
         __make_index_sequence<
           __sum<
             0u,
             std::tuple_size<typename std::decay<Tuples>::type>::value...
           >::value
         >(),
         f,
         std::forward<Tuples>(ts)...
       )
     )
{
  const size_t N = __sum<0u, std::tuple_size<typename std::decay<Tuples>::type>::value...>::value;
  return __tuple_cat_apply_impl(__make_index_sequence<N>(), f, std::forward<Tuples>(ts)...);
}


template<class Function, class Tuple, size_t... I>
TUPLE_UTILITY_ANNOTATION
auto __tuple_apply_impl(Function f, Tuple&& t, __index_sequence<I...>)
  -> decltype(
       f(__get<I>(std::forward<Tuple>(t))...)
     )
{
  return f(__get<I>(std::forward<Tuple>(t))...);
}



template<class Function, class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_apply(Function f, Tuple&& t)
  -> decltype(
       tuple_cat_apply(f, std::forward<Tuple>(t))
     )
{
  return tuple_cat_apply(f, std::forward<Tuple>(t));
}


template<class... Tuples>
TUPLE_UTILITY_ANNOTATION
auto tuple_zip(Tuples&&... tuples)
  -> decltype(
       tuple_map(__std_tuple_maker{}, std::forward<Tuples>(tuples)...)
     )
{
  return tuple_map(__std_tuple_maker{}, std::forward<Tuples>(tuples)...);
}


// concatenate two index_sequences
template<class IndexSequence1, class IndexSequence2> struct __index_sequence_cat_impl;


template<size_t... Indices1, size_t... Indices2>
struct __index_sequence_cat_impl<__index_sequence<Indices1...>, __index_sequence<Indices2...>>
{
  using type = __index_sequence<Indices1..., Indices2...>;
};

template<class IndexSequence1, class IndexSequence2>
using __index_sequence_cat = typename __index_sequence_cat_impl<IndexSequence1,IndexSequence2>::type;


template<template<size_t> class MetaFunction, class Indices>
struct __filter_index_sequence_impl;


// an empty sequence filters to the empty sequence
template<template<size_t> class MetaFunction>
struct __filter_index_sequence_impl<MetaFunction, __index_sequence<>>
{
  using type = __index_sequence<>;
};

template<template<size_t> class MetaFunction, size_t Index0, size_t... Indices>
struct __filter_index_sequence_impl<MetaFunction, __index_sequence<Index0, Indices...>>
{
  // recurse and filter the rest of the indices
  using rest = typename __filter_index_sequence_impl<MetaFunction,__index_sequence<Indices...>>::type;

  // concatenate Index0 with rest if Index0 passes the filter
  // else, just return rest
  using type = typename std::conditional<
    MetaFunction<Index0>::value,
    __index_sequence_cat<
      __index_sequence<Index0>,
      rest
    >,
    rest
  >::type;
};


template<template<size_t> class MetaFunction, class Indices>
using __filter_index_sequence = typename __filter_index_sequence_impl<MetaFunction,Indices>::type;


template<template<class> class MetaFunction, class Tuple>
struct __index_filter
{
  using traits = tuple_traits<Tuple>;

  template<size_t i>
  using filter = MetaFunction<typename traits::template element_type<i>>;
};


template<template<class> class MetaFunction, class Tuple>
using __make_filtered_indices_for_tuple =
  __filter_index_sequence<
    __index_filter<MetaFunction, Tuple>::template filter,
    __make_index_sequence<tuple_traits<Tuple>::size>
  >;


// XXX nvcc 7.0 has trouble with this template template parameter
//template<template<class> class MetaFunction, class Tuple, class Function>
//TUPLE_UTILITY_ANNOTATION
//auto tuple_filter_invoke(Tuple&& t, Function f)
//  -> decltype(
//       __tuple_apply_impl(
//         f,
//         std::forward<Tuple>(t),
//         __make_filtered_indices_for_tuple<MetaFunction, typename std::decay<Tuple>::type>{}
//       )
//     )
//{
//  using filtered_indices = __make_filtered_indices_for_tuple<MetaFunction, typename std::decay<Tuple>::type>;
//
//  return __tuple_apply_impl(f, std::forward<Tuple>(t), filtered_indices{});
//}
template<template<class> class MetaFunction, class Tuple, class Function, class Indices = __make_filtered_indices_for_tuple<MetaFunction, typename std::decay<Tuple>::type>>
TUPLE_UTILITY_ANNOTATION
auto tuple_filter_invoke(Tuple&& t, Function f)
  -> decltype(
       __tuple_apply_impl(
         f,
         std::forward<Tuple>(t),
         Indices{}
       )
     )
{
  return __tuple_apply_impl(f, std::forward<Tuple>(t), Indices{});
}


template<template<class> class MetaFunction, class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_filter(Tuple&& t)
  -> decltype(
       tuple_filter_invoke<MetaFunction>(std::forward<Tuple>(t), __std_tuple_maker{})
     )
{
  return tuple_filter_invoke<MetaFunction>(std::forward<Tuple>(t), __std_tuple_maker{});
}


#ifdef TUPLE_UTILITY_NAMESPACE
} // close namespace
#endif // TUPLE_UTILITY_NAMESPACE


#ifdef TUPLE_UTILITY_ANNOTATION_NEEDS_UNDEF
#undef TUPLE_UTILITY_ANNOTATION
#undef TUPLE_UTILITY_ANNOTATION_NEEDS_UNDEF
#endif

