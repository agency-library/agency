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

#define TUPLE_UTILITY_NAMESPACE __tu

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


template<class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_head(Tuple&& t)
  -> decltype(
       std::get<0>(std::forward<Tuple>(t))
     )
{
  return std::get<0>(std::forward<Tuple>(t));
}


template<class... Args>
TUPLE_UTILITY_ANNOTATION
auto __forward_tuple_tail_impl_impl(Args&&... args)
  -> decltype(
       std::forward_as_tuple(args...)
     )
{
  return std::forward_as_tuple(args...);
}

template<class Tuple, size_t... I>
TUPLE_UTILITY_ANNOTATION
auto __forward_tuple_tail_impl(Tuple&& t, __index_sequence<I...>)
  -> decltype(
       __forward_tuple_tail_impl_impl(std::get<I+1>(std::forward<Tuple>(t))...)
     )
{
  return __forward_tuple_tail_impl_impl(std::get<I+1>(std::forward<Tuple>(t))...);
}


// forward_tuple_tail() returns t's tail as a tuple of references
template<class Tuple>
TUPLE_UTILITY_ANNOTATION
auto forward_tuple_tail(typename std::remove_reference<Tuple>::type& t)
  -> decltype(
       __forward_tuple_tail_impl(
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
  return __forward_tuple_tail_impl(std::forward<Tuple>(t), indices());
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
       __tuple_tail_impl_impl(std::get<I+1>(std::forward<Tuple>(t))...)
     )
{
  return __tuple_tail_impl_impl(std::get<I+1>(std::forward<Tuple>(t))...);
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


template<size_t... I, class Tuple>
TUPLE_UTILITY_ANNOTATION
auto __tuple_take_impl(Tuple&& t, __index_sequence<I...>)
  -> decltype(
       std::make_tuple(
         std::get<I>(std::forward<Tuple>(t))...
       )
     )
{
  return std::make_tuple(std::get<I>(std::forward<Tuple>(t))...);
}


template<size_t N, class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_take(Tuple&& t)
  -> decltype(
       __tuple_take_impl(std::forward<Tuple>(t), __make_index_sequence<N>())
     )
{
  return __tuple_take_impl(std::forward<Tuple>(t), __make_index_sequence<N>());
}


template<size_t N, class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_drop(Tuple&& t)
  -> decltype(
       tuple_take<std::tuple_size<typename std::decay<Tuple>::type>::value - N>(std::forward<Tuple>(t))
     )
{
  return tuple_take<std::tuple_size<typename std::decay<Tuple>::type>::value - N>(std::forward<Tuple>(t));
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
       std::get<
         std::tuple_size<
           typename std::decay<Tuple>::type
         >::value - 1
       >(std::forward<Tuple>(t))
     )
{
  const size_t i = std::tuple_size<typename std::decay<Tuple>::type>::value - 1;
  return std::get<i>(std::forward<Tuple>(t));
}


template<class Tuple, class T, size_t... I>
TUPLE_UTILITY_ANNOTATION
auto __tuple_append_impl(Tuple&& t, T&& x, __index_sequence<I...>)
  -> decltype(
       std::make_tuple(std::get<I>(std::forward<Tuple>(t))..., std::forward<T>(x))
     )
{
  return std::make_tuple(std::get<I>(std::forward<Tuple>(t))..., std::forward<T>(x));
}


template<class Tuple, class T>
TUPLE_UTILITY_ANNOTATION
auto tuple_append(Tuple&& t, T&& x)
  -> decltype(
       __tuple_append_impl(
         std::forward<Tuple>(t),
         std::forward<T>(x),
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
  return __tuple_append_impl(std::forward<Tuple>(t), std::forward<T>(x), indices());
}



template<size_t I, typename Function, typename... Tuples>
TUPLE_UTILITY_ANNOTATION
auto __tuple_map_invoke(Function f, Tuples&&... ts)
  -> decltype(
       f(std::get<I>(std::forward<Tuples>(ts))...)
     )
{
  return f(std::get<I>(std::forward<Tuples>(ts))...);
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
  return T{std::get<I>(t)...};
}


template<class T, class Tuple>
TUPLE_UTILITY_ANNOTATION
T make_from_tuple(const Tuple& t)
{
  return make_from_tuple_impl<T>(t, __make_index_sequence<std::tuple_size<Tuple>::value>());
}


template<class Tuple, class T, class Function,
         class = typename std::enable_if<
           (std::tuple_size<
             typename std::decay<Tuple>::type
           >::value == 0)
         >::type>
TUPLE_UTILITY_ANNOTATION
T tuple_reduce(Tuple&&, T init, Function)
{
  return init;
}


template<class Tuple, class T, class Function>
TUPLE_UTILITY_ANNOTATION
T tuple_reduce(Tuple&& t, T init, Function f,
               typename std::enable_if<
                 (std::tuple_size<
                   typename std::decay<Tuple>::type
                 >::value > 0)
               >::type* = 0)
{
  return f(
    tuple_head(std::forward<Tuple>(t)),
    tuple_reduce(forward_tuple_tail<Tuple>(t), init, f)
  );
}


template<class Function, class Tuple1, class... Tuples>
TUPLE_UTILITY_ANNOTATION
typename std::enable_if<
  (std::tuple_size<__decay_t<Tuple1>>::value == 0)
>::type
  tuple_for_each(Function f, Tuple1&& t1, Tuples&&... ts)
{
  return;
}


template<class Function, class Tuple1, class... Tuples>
TUPLE_UTILITY_ANNOTATION
typename std::enable_if<
  (std::tuple_size<__decay_t<Tuple1>>::value > 0)
>::type
  tuple_for_each(Function f, Tuple1&& t1, Tuples&&... ts)
{
  f(tuple_head(std::forward<Tuple1>(t1)), tuple_head(std::forward<Tuples>(ts))...);
  return tuple_for_each(f, forward_tuple_tail<Tuple1>(t1), forward_tuple_tail<Tuples>(ts)...);
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
  os << std::get<0>(t);
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


template<class Function, class Tuple, size_t... I>
TUPLE_UTILITY_ANNOTATION
auto __tuple_apply_impl(Function f, Tuple&& t, __index_sequence<I...>)
  -> decltype(
       f(std::get<I>(std::forward<Tuple>(t))...)
     )
{
  return f(std::get<I>(std::forward<Tuple>(t))...);
}



template<class Function, class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_apply(Function f, Tuple&& t)
  -> decltype(
       __tuple_apply_impl(
         f,
         std::forward<Tuple>(t),
         __make_index_sequence<std::tuple_size<__decay_t<Tuple>>::value>()
       )
     )
{
  using indices = __make_index_sequence<std::tuple_size<__decay_t<Tuple>>::value>;
  return __tuple_apply_impl(f, std::forward<Tuple>(t), indices());
}


template<class Function, class... StdTuples>
TUPLE_UTILITY_ANNOTATION
auto __tuple_cat_apply_impl(Function f, StdTuples&&... tuples)
  -> decltype(
       tuple_apply(f, std::tuple_cat(std::forward<StdTuples>(tuples)...))
     )
{
  return tuple_apply(f, std::tuple_cat(std::forward<StdTuples>(tuples)...));
}


template<class Function, class... Tuples>
TUPLE_UTILITY_ANNOTATION
auto tuple_cat_apply(Function f, Tuples&&... tuples)
  -> decltype(
       __tuple_cat_apply_impl(f, tuple_apply(__std_tuple_maker{}, std::forward<Tuples>(tuples))...)
     )
{
  // transform each tuple into a std::tuple with tuple_apply
  // then call __tuple_cat_apply_impl
  return __tuple_cat_apply_impl(f, tuple_apply(__std_tuple_maker{}, std::forward<Tuples>(tuples))...);
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


#ifdef TUPLE_UTILITY_NAMESPACE
} // close namespace
#endif // TUPLE_UTILITY_NAMESPACE


#ifdef TUPLE_UTILITY_ANNOTATION_NEEDS_UNDEF
#undef TUPLE_UTILITY_ANNOTATION
#undef TUPLE_UTILITY_ANNOTATION_NEEDS_UNDEF
#endif

