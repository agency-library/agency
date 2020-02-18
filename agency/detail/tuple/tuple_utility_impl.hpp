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


#define TUPLE_UTILITY_REQUIRES(...) typename std::enable_if<(__VA_ARGS__)>::type* = nullptr


// figure out what namespace to put everything in
#if !defined(TUPLE_UTILITY_NAMESPACE)
#  if defined(TUPLE_UTILITY_NAMESPACE_OPEN_BRACE) or defined(TUPLE_UTILITY_NAMESPACE_CLOSE_BRACE)
#    error "Either all of TUPLE_UTILITY_NAMESPACE, TUPLE_UTILITY_NAMESPACE_OPEN_BRACE, and TUPLE_UTILITY_NAMESPACE_CLOSE_BRACE must be defined, or none of them."
#  endif

// by default, there is no namespace
# define TUPLE_UTILITY_NAMESPACE
# define TUPLE_UTILITY_NAMESPACE_OPEN_BRACE
# define TUPLE_UTILITY_NAMESPACE_CLOSE_BRACE
# define TUPLE_UTILITY_NAMESPACE_NEEDS_UNDEF

#elif defined(TUPLE_UTILITY_NAMESPACE) or defined(TUPLE_UTILITY_NAMESPACE_OPEN_BRACE) or defined(TUPLE_UTILITY_NAMESPACE_CLOSE_BRACE)
#  if !defined(TUPLE_UTILITY_NAMESPACE) or !defined(TUPLE_UTILITY_NAMESPACE_OPEN_BRACE) or !defined(TUPLE_UTILITY_NAMESPACE_CLOSE_BRACE)
#    error "Either all of TUPLE_UTILITY_NAMESPACE, TUPLE_UTILITY_NAMESPACE_OPEN_BRACE, and TUPLE_UTILITY_NAMESPACE_CLOSE_BRACE must be defined, or none of them."
#  endif
#endif


TUPLE_UTILITY_NAMESPACE_OPEN_BRACE


namespace detail
{


template<class T>
using decay_t = typename std::decay<T>::type;


template<class... Conditions>
struct conjunction;

template<>
struct conjunction<> : std::true_type {};

template<class Condition, class... Conditions>
struct conjunction<Condition, Conditions...>
  : std::integral_constant<
      bool,
      Condition::value and conjunction<Conditions...>::value
    >
{};


// define index sequence in case it is missing
template<size_t... I> struct index_sequence {};

template<size_t Start, typename Indices, size_t End>
struct make_index_sequence_impl;

template<size_t Start, size_t... Indices, size_t End>
struct make_index_sequence_impl<
  Start,
  index_sequence<Indices...>, 
  End
>
{
  typedef typename make_index_sequence_impl<
    Start + 1,
    index_sequence<Indices..., Start>,
    End
  >::type type;
};

template<size_t End, size_t... Indices>
struct make_index_sequence_impl<End, index_sequence<Indices...>, End>
{
  typedef index_sequence<Indices...> type;
};

template<size_t N>
using make_index_sequence = typename make_index_sequence_impl<0, index_sequence<>, N>::type;



template<class Reference, std::size_t i>
struct has_get_member_function_impl
{
  // XXX consider requiring that Result matches the expected result,
  //     i.e. propagate_reference_t<Reference, std::tuple_element_t<i,T>>
  template<class T = Reference,
           std::size_t j = i,
           class Result = decltype(std::declval<T>().template get<j>())
          >
  static std::true_type test(int);

  static std::false_type test(...);

  using type = decltype(test(0));
};

template<class Reference, std::size_t i>
using has_get_member_function = typename has_get_member_function_impl<Reference,i>::type;


template<class Reference, std::size_t i>
struct has_std_get_free_function_impl
{
  // XXX consider requiring that Result matches the expected result,
  //     i.e. propagate_reference_t<Reference, std::tuple_element_t<i,T>>
  template<class T = Reference,
           std::size_t j = i,
           class Result = decltype(std::get<j>(std::declval<T>()))
          >
  static std::true_type test(int);

  static std::false_type test(...);

  using type = decltype(test(0));
};

template<class T, std::size_t i>
using has_std_get_free_function = typename has_std_get_free_function_impl<T,i>::type;


template<class Reference>
struct has_operator_bracket_impl
{
  // XXX consider requiring that Result matches the expected result,
  //     i.e. propagate_reference_t<Reference, std::tuple_element_t<i,T>>
  template<class T = Reference,
           class Result = decltype(std::declval<T>()[0])
          >
  static std::true_type test(int);

  static std::false_type test(...);

  using type = decltype(test(0));
};

template<class T>
using has_operator_bracket = typename has_operator_bracket_impl<T>::type;


} // end detail


template<class T>
struct tuple_traits
{
  using tuple_type = T;

  static const size_t size = std::tuple_size<tuple_type>::value; 

  template<size_t i>
  using element_type = typename std::tuple_element<i,tuple_type>::type;

  // these overloads of get use member t.template get<i>()
  template<size_t i,
           TUPLE_UTILITY_REQUIRES(
             detail::has_get_member_function<tuple_type&, i>::value
           )>
  TUPLE_UTILITY_ANNOTATION
  static element_type<i>& get(tuple_type& t)
  {
    return t.template get<i>();
  }

  template<size_t i,
           TUPLE_UTILITY_REQUIRES(
             detail::has_get_member_function<const tuple_type&, i>::value
           )>
  TUPLE_UTILITY_ANNOTATION
  static const element_type<i>& get(const tuple_type& t)
  {
    return t.template get<i>();
  }

  template<size_t i,
           TUPLE_UTILITY_REQUIRES(
             detail::has_get_member_function<tuple_type&&, i>::value
           )>
  TUPLE_UTILITY_ANNOTATION
  static element_type<i>&& get(tuple_type&& t)
  {
    return std::move(t).template get<i>();
  }


  // these overloads of get use std::get<i>(t)
  template<size_t i,
           TUPLE_UTILITY_REQUIRES(
             !detail::has_get_member_function<tuple_type&, i>::value and
             detail::has_std_get_free_function<tuple_type&, i>::value
           )>
  TUPLE_UTILITY_ANNOTATION
  static element_type<i>& get(tuple_type& t)
  {
    return std::get<i>(t);
  }

  template<size_t i,
           TUPLE_UTILITY_REQUIRES(
             !detail::has_get_member_function<const tuple_type&, i>::value and
             detail::has_std_get_free_function<const tuple_type&, i>::value
           )>
  TUPLE_UTILITY_ANNOTATION
  static const element_type<i>& get(const tuple_type& t)
  {
    return std::get<i>(t);
  }

  template<size_t i,
           TUPLE_UTILITY_REQUIRES(
             !detail::has_get_member_function<tuple_type&&, i>::value and
             detail::has_std_get_free_function<tuple_type&&, i>::value
           )>
  TUPLE_UTILITY_ANNOTATION
  static element_type<i>&& get(tuple_type&& t)
  {
    return std::get<i>(std::move(t));
  }


  // these overloads of get use t[i] (operator bracket)
  template<size_t i,
           TUPLE_UTILITY_REQUIRES(
             !detail::has_get_member_function<tuple_type&, i>::value and
             !detail::has_std_get_free_function<tuple_type&, i>::value and
             detail::has_operator_bracket<tuple_type&>::value
           )>
  TUPLE_UTILITY_ANNOTATION
  static element_type<i>& get(tuple_type& t)
  {
    return t[i];
  }

  template<size_t i,
           TUPLE_UTILITY_REQUIRES(
             !detail::has_get_member_function<tuple_type&, i>::value and
             !detail::has_std_get_free_function<tuple_type&, i>::value and
             detail::has_operator_bracket<tuple_type&>::value
           )>
  TUPLE_UTILITY_ANNOTATION
  static const element_type<i>& get(const tuple_type& t)
  {
    return t[i];
  }

  template<size_t i,
           TUPLE_UTILITY_REQUIRES(
             !detail::has_get_member_function<tuple_type&, i>::value and
             !detail::has_std_get_free_function<tuple_type&, i>::value and
             detail::has_operator_bracket<tuple_type&>::value
           )>
  TUPLE_UTILITY_ANNOTATION
  static element_type<i>&& get(tuple_type&& t)
  {
    return std::move(std::move(t)[i]);
  }
};


template<class T>
struct is_tuple_like
{
  // XXX this should also check for the existence of std::tuple_element<T,I>::type
  //     for all of T's elements
  template<class U = T,
           std::size_t = std::tuple_size<U>::value
          >
  static constexpr bool test(int)
  {
    return true;
  }

  template<class>
  static constexpr bool test(...)
  {
    return false;
  }

  static constexpr bool value = test<T>(0);
};


static_assert(is_tuple_like<std::tuple<int,int>>::value, "std::tuple should be tuple-like.");


template<class T, class... Types>
struct tuple_rebind;


// tuple-like templates can always be rebound
template<template<class...> class TupleTemplate, class... OriginalTypes, class... ReboundTypes>
struct tuple_rebind<TupleTemplate<OriginalTypes...>, ReboundTypes...>
{
  using type = TupleTemplate<ReboundTypes...>;
};


// array-like templates can be rebound if all of the types to use in the rebinding are the same
template<template<class,std::size_t> class ArrayTemplate, class OriginalType, std::size_t N, class ReboundType, class... ReboundTypes>
struct tuple_rebind<ArrayTemplate<OriginalType,N>, ReboundType, ReboundTypes...>
  : std::conditional<
      detail::conjunction<std::is_same<ReboundType,ReboundTypes>...>::value, // if all of the types to rebind are the same,
      ArrayTemplate<ReboundType, 1 + sizeof...(ReboundTypes)>,               // then reinstantiate the ArrayTemplate using ReboundType
      std::enable_if<false>                                                  // otherwise, do not define a member named type
    >::type
{};


// pair-like templates can be rebound if we have two types to use in the rebinding
template<template<class,class> class PairTemplate, class OriginalType1, class OriginalType2, class ReboundType1, class ReboundType2>
struct tuple_rebind<PairTemplate<OriginalType1,OriginalType2>, ReboundType1, ReboundType2>
{
  using type = PairTemplate<ReboundType1, ReboundType2>;
};


template<class Tuple, class... Types>
using tuple_rebind_t = typename tuple_rebind<Tuple,Types...>::type;


template<size_t i, class Tuple>
TUPLE_UTILITY_ANNOTATION
auto get(Tuple&& t)
  -> decltype(
       tuple_traits<TUPLE_UTILITY_NAMESPACE::detail::decay_t<Tuple>>::template get<i>(std::forward<Tuple>(t))
     )
{
  return tuple_traits<TUPLE_UTILITY_NAMESPACE::detail::decay_t<Tuple>>::template get<i>(std::forward<Tuple>(t));
}


template<class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_head(Tuple&& t)
  -> decltype(
       TUPLE_UTILITY_NAMESPACE::get<0>(std::forward<Tuple>(t))
     )
{
  return TUPLE_UTILITY_NAMESPACE::get<0>(std::forward<Tuple>(t));
}


namespace detail
{


template<class Tuple, class Function, size_t... I>
TUPLE_UTILITY_ANNOTATION
auto tuple_tail_invoke_impl(Tuple&& t, Function f, index_sequence<I...>)
  -> decltype(
       f(
         TUPLE_UTILITY_NAMESPACE::get<I+1>(std::forward<Tuple>(t))...
       )
     )
{
  return f(TUPLE_UTILITY_NAMESPACE::get<I+1>(std::forward<Tuple>(t))...);
}


} // end detail


template<class Tuple, class Function>
TUPLE_UTILITY_ANNOTATION
auto tuple_tail_invoke(Tuple&& t, Function f)
  -> decltype(
       detail::tuple_tail_invoke_impl(
         std::forward<Tuple>(t),
         f,
         detail::make_index_sequence<
           std::tuple_size<
             typename std::decay<Tuple>::type
           >::value - 1
         >()
       )
     )
{
  return tuple_tail_invoke_impl(
    std::forward<Tuple>(t),
    f,
    detail::make_index_sequence<
      std::tuple_size<
        typename std::decay<Tuple>::type
      >::value - 1
    >()
  );
}


namespace detail
{


template<class Tuple, class Function, size_t... I>
TUPLE_UTILITY_ANNOTATION
auto tuple_prefix_invoke_impl(Tuple&& t, Function f, index_sequence<I...>)
  -> decltype(
       f(
         TUPLE_UTILITY_NAMESPACE::get<I>(std::forward<Tuple>(t))...
       )
     )
{
  return f(TUPLE_UTILITY_NAMESPACE::get<I>(std::forward<Tuple>(t))...);
}


} // end detail


template<class Tuple, class Function>
TUPLE_UTILITY_ANNOTATION
auto tuple_prefix_invoke(Tuple&& t, Function f)
  -> decltype(
       detail::tuple_prefix_invoke_impl(
         std::forward<Tuple>(t),
         f,
         detail::make_index_sequence<
           std::tuple_size<
             typename std::decay<Tuple>::type
           >::value - 1
         >()
       )
     )
{
  return detail::tuple_prefix_invoke_impl(
    std::forward<Tuple>(t),
    f,
    detail::make_index_sequence<
      std::tuple_size<
        typename std::decay<Tuple>::type
      >::value - 1
    >()
  );
}


namespace detail
{


struct tuple_forwarder
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


} // end detail


// forward_tuple_tail() returns t's tail as a tuple of references
template<class Tuple>
TUPLE_UTILITY_ANNOTATION
auto forward_tuple_tail(typename std::remove_reference<Tuple>::type& t)
  -> decltype(
       tuple_tail_invoke(std::forward<Tuple>(t), detail::tuple_forwarder{})
     )
{
  return tuple_tail_invoke(std::forward<Tuple>(t), detail::tuple_forwarder{});
}


namespace detail
{


template<class... Args>
TUPLE_UTILITY_ANNOTATION
auto tuple_tail_impl_impl(Args&&... args)
  -> decltype(
       std::make_tuple(args...)
     )
{
  return std::make_tuple(args...);
}

template<class Tuple, size_t... I>
TUPLE_UTILITY_ANNOTATION
auto tuple_tail_impl(Tuple&& t, index_sequence<I...>)
  -> decltype(
       detail::tuple_tail_impl_impl(TUPLE_UTILITY_NAMESPACE::get<I+1>(std::forward<Tuple>(t))...)
     )
{
  return detail::tuple_tail_impl_impl(TUPLE_UTILITY_NAMESPACE::get<I+1>(std::forward<Tuple>(t))...);
}


} // end detail


// tuple_tail() returns t's tail as a tuple of values
// i.e., it makes a copy of t's tail
template<class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_tail(Tuple&& t)
  -> decltype(
       detail::tuple_tail_impl(
         std::forward<Tuple>(t),
         detail::make_index_sequence<
           std::tuple_size<
             typename std::decay<Tuple>::type
           >::value - 1
         >()
       )
     )
{
  using indices = detail::make_index_sequence<
    std::tuple_size<
      typename std::decay<Tuple>::type
    >::value - 1
  >;
  return detail::tuple_tail_impl(std::forward<Tuple>(t), indices());
}


namespace detail
{


template<class Tuple, class Function, size_t... I>
TUPLE_UTILITY_ANNOTATION
auto tuple_take_invoke_impl(Tuple&& t, Function f, index_sequence<I...>)
  -> decltype(
       f(
         TUPLE_UTILITY_NAMESPACE::get<I>(std::forward<Tuple>(t))...
       )
     )
{
  return f(TUPLE_UTILITY_NAMESPACE::get<I>(std::forward<Tuple>(t))...);
}


} // end detail


template<size_t N, class Tuple, class Function>
TUPLE_UTILITY_ANNOTATION
auto tuple_take_invoke(Tuple&& t, Function f)
  -> decltype(
       detail::tuple_take_invoke_impl(std::forward<Tuple>(t), f, detail::make_index_sequence<N>())
     )
{
  return detail::tuple_take_invoke_impl(std::forward<Tuple>(t), f, detail::make_index_sequence<N>());
}


namespace detail
{


struct std_tuple_factory
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


} // end detail


template<size_t N, class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_take(Tuple&& t)
  -> decltype(
       tuple_take_invoke<N>(std::forward<Tuple>(t), detail::std_tuple_factory())
     )
{
  return tuple_take_invoke<N>(std::forward<Tuple>(t), detail::std_tuple_factory());
}


namespace detail
{


template<size_t N, size_t... I, class Tuple, class Function>
TUPLE_UTILITY_ANNOTATION
auto tuple_drop_invoke_impl(Tuple&& t, Function f, index_sequence<I...>)
  -> decltype(
       f(TUPLE_UTILITY_NAMESPACE::get<N + I>(std::forward<Tuple>(t))...)
     )
{
  return f(TUPLE_UTILITY_NAMESPACE::get<N + I>(std::forward<Tuple>(t))...);
}


} // end detail


template<size_t N, class Tuple, class Function>
TUPLE_UTILITY_ANNOTATION
auto tuple_drop_invoke(Tuple&& t, Function f)
  -> decltype(
       detail::tuple_drop_invoke_impl<N>(
         std::forward<Tuple>(t),
         f,
         detail::make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value - N>()
       )
     )
{
  return detail::tuple_drop_invoke_impl<N>(
    std::forward<Tuple>(t),
    f,
    detail::make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value - N>()
  );
}


template<size_t N, class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_drop(Tuple&& t)
  -> decltype(
       tuple_drop_invoke<N>(std::forward<Tuple>(t), detail::std_tuple_factory())
     )
{
  return TUPLE_UTILITY_NAMESPACE::tuple_drop_invoke<N>(std::forward<Tuple>(t), detail::std_tuple_factory());
}


template<size_t N, class Tuple, class Function>
TUPLE_UTILITY_ANNOTATION
auto tuple_drop_back_invoke(Tuple&& t, Function f)
  -> decltype(
       tuple_take_invoke<std::tuple_size<typename std::decay<Tuple>::type>::value - N>(std::forward<Tuple>(t), f)
     )
{
  return tuple_take_invoke<std::tuple_size<typename std::decay<Tuple>::type>::value - N>(std::forward<Tuple>(t), f);
}


template<size_t N, class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_drop_back(Tuple&& t)
  -> decltype(
       tuple_drop_back_invoke<N>(std::forward<Tuple>(t), detail::std_tuple_factory())
     )
{
  return tuple_drop_back_invoke<N>(std::forward<Tuple>(t), detail::std_tuple_factory());
}


template<class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_drop_last(Tuple&& t)
  -> decltype(
       TUPLE_UTILITY_NAMESPACE::tuple_drop_back<1>(std::forward<Tuple>(t))
     )
{
  return TUPLE_UTILITY_NAMESPACE::tuple_drop_back<1>(std::forward<Tuple>(t));
}


template<class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_last(Tuple&& t)
  -> decltype(
       TUPLE_UTILITY_NAMESPACE::get<
         std::tuple_size<
           typename std::decay<Tuple>::type
         >::value - 1
       >(std::forward<Tuple>(t))
     )
{
  const size_t i = std::tuple_size<typename std::decay<Tuple>::type>::value - 1;
  return TUPLE_UTILITY_NAMESPACE::get<i>(std::forward<Tuple>(t));
}


namespace detail
{


template<class Tuple, class T, class Function, size_t... I>
TUPLE_UTILITY_ANNOTATION
auto tuple_append_invoke_impl(Tuple&& t, T&& x, Function f, index_sequence<I...>)
  -> decltype(
       f(TUPLE_UTILITY_NAMESPACE::get<I>(std::forward<Tuple>(t))..., std::forward<T>(x))
     )
{
  return f(TUPLE_UTILITY_NAMESPACE::get<I>(std::forward<Tuple>(t))..., std::forward<T>(x));
}


} // end detail


template<class Tuple, class T, class Function>
TUPLE_UTILITY_ANNOTATION
auto tuple_append_invoke(Tuple&& t, T&& x, Function f)
  -> decltype(
       detail::tuple_append_invoke_impl(
         std::forward<Tuple>(t),
         std::forward<T>(x),
         f,
         detail::make_index_sequence<
           std::tuple_size<
             typename std::decay<Tuple>::type
           >::value
         >()
       )
     )
{
  using indices = detail::make_index_sequence<
    std::tuple_size<typename std::decay<Tuple>::type>::value
  >;
  return detail::tuple_append_invoke_impl(std::forward<Tuple>(t), std::forward<T>(x), f, indices());
}


namespace detail
{


template<class Tuple, class T, class Function, size_t... I>
TUPLE_UTILITY_ANNOTATION
auto tuple_prepend_invoke_impl(Tuple&& t, T&& x, Function f, index_sequence<I...>)
  -> decltype(
       f(std::forward<T>(x), TUPLE_UTILITY_NAMESPACE::get<I>(std::forward<Tuple>(t))...)
     )
{
  return f(std::forward<T>(x), TUPLE_UTILITY_NAMESPACE::get<I>(std::forward<Tuple>(t))...);
}


} // end detail


template<class Tuple, class T, class Function>
TUPLE_UTILITY_ANNOTATION
auto tuple_prepend_invoke(Tuple&& t, T&& x, Function f)
  -> decltype(
       detail::tuple_prepend_invoke_impl(
         std::forward<Tuple>(t),
         std::forward<T>(x),
         f,
         detail::make_index_sequence<
           std::tuple_size<
             typename std::decay<Tuple>::type
           >::value
         >()
       )
     )
{
  using indices = detail::make_index_sequence<
    std::tuple_size<typename std::decay<Tuple>::type>::value
  >;
  return detail::tuple_prepend_invoke_impl(std::forward<Tuple>(t), std::forward<T>(x), f, indices());
}


template<class Tuple, class T>
TUPLE_UTILITY_ANNOTATION
auto tuple_append(Tuple&& t, T&& x)
  -> decltype(
       tuple_append_invoke(std::forward<Tuple>(t), std::forward<T>(x), detail::std_tuple_factory())
     )
{
  return tuple_append_invoke(std::forward<Tuple>(t), std::forward<T>(x), detail::std_tuple_factory());
}


template<class Tuple, class T>
TUPLE_UTILITY_ANNOTATION
auto tuple_prepend(Tuple&& t, T&& x)
  -> decltype(
       tuple_prepend_invoke(std::forward<Tuple>(t), std::forward<T>(x), detail::std_tuple_factory())
     )
{
  return tuple_prepend_invoke(std::forward<Tuple>(t), std::forward<T>(x), detail::std_tuple_factory());
}


namespace detail
{


template<size_t I, typename Function, typename... Tuples>
TUPLE_UTILITY_ANNOTATION
auto tuple_map_invoke(Function f, Tuples&&... ts)
  -> decltype(
       f(TUPLE_UTILITY_NAMESPACE::get<I>(std::forward<Tuples>(ts))...)
     )
{
  return f(TUPLE_UTILITY_NAMESPACE::get<I>(std::forward<Tuples>(ts))...);
}


template<size_t... I, typename Function1, typename Function2, typename... Tuples>
TUPLE_UTILITY_ANNOTATION
auto tuple_map_with_make_impl(index_sequence<I...>, Function1 f, Function2 make, Tuples&&... ts)
  -> decltype(
       make(
         tuple_map_invoke<I>(f, std::forward<Tuples>(ts)...)...
       )
     )
{
  return make(
    tuple_map_invoke<I>(f, std::forward<Tuples>(ts)...)...
  );
}


} // end detail


template<typename Function1, typename Function2, typename Tuple, typename... Tuples>
TUPLE_UTILITY_ANNOTATION
auto tuple_map_with_make(Function1 f, Function2 make, Tuple&& t, Tuples&&... ts)
  -> decltype(
       detail::tuple_map_with_make_impl(
         detail::make_index_sequence<
           std::tuple_size<detail::decay_t<Tuple>>::value
         >(),
         f,
         make,
         std::forward<Tuple>(t),
         std::forward<Tuples>(ts)...
       )
     )
{
  return detail::tuple_map_with_make_impl(
    detail::make_index_sequence<
      std::tuple_size<detail::decay_t<Tuple>>::value
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
       tuple_map_with_make(f, detail::std_tuple_factory(), std::forward<Tuple>(t), std::forward<Tuples>(ts)...)
     )
{
  return tuple_map_with_make(f, detail::std_tuple_factory(), std::forward<Tuple>(t), std::forward<Tuples>(ts)...);
}


namespace detail
{


template<class T, class Tuple, size_t... I>
TUPLE_UTILITY_ANNOTATION
T make_from_tuple_impl(const Tuple& t, index_sequence<I...>)
{
  // use constructor syntax
  return T(TUPLE_UTILITY_NAMESPACE::get<I>(t)...);
}


} // end detail


template<class T, class Tuple>
TUPLE_UTILITY_ANNOTATION
T make_from_tuple(const Tuple& t)
{
  return detail::make_from_tuple_impl<T>(t, detail::make_index_sequence<std::tuple_size<Tuple>::value>());
}


namespace detail
{


template<class Function>
struct tuple_for_each_functor
{
  mutable Function f;
  
  template<class... Args>
  TUPLE_UTILITY_ANNOTATION
  int operator()(Args&&... args) const
  {
    f(std::forward<Args>(args)...);
    return 0;
  }
};


template<size_t I, class Function, class... Tuples>
TUPLE_UTILITY_ANNOTATION
int apply_row(Function f, Tuples&&... ts)
{
  f(TUPLE_UTILITY_NAMESPACE::get<I>(std::forward<Tuples>(ts))...);
  return 0;
}


template<class... Args>
TUPLE_UTILITY_ANNOTATION
void swallow(Args&&...) {}


template<size_t... Indices, class Function, class... Tuples>
TUPLE_UTILITY_ANNOTATION
void tuple_for_each_impl(index_sequence<Indices...>, Function f, Tuples&&... ts)
{
  auto g = tuple_for_each_functor<Function>{f};

  // XXX swallow g to WAR nvcc 7.0 unused variable warning
  detail::swallow(g);

  // unpacking into a init lists preserves the order given by Indices
  using ints = int[];
  (void) ints{0, ((detail::apply_row<Indices>(g, std::forward<Tuples>(ts)...)), void(), 0)...};
}


} // end detail


template<size_t N, class Function, class Tuple1, class... Tuples>
TUPLE_UTILITY_ANNOTATION
void tuple_for_each_n(Function f, Tuple1&& t1, Tuples&&... ts)
{
  detail::tuple_for_each_impl(detail::make_index_sequence<N>(), f, std::forward<Tuple1>(t1), std::forward<Tuples>(ts)...);
}


template<class Function, class Tuple1, class... Tuples>
TUPLE_UTILITY_ANNOTATION
void tuple_for_each(Function f, Tuple1&& t1, Tuples&&... ts)
{
  tuple_for_each_n<std::tuple_size<detail::decay_t<Tuple1>>::value>(f, std::forward<Tuple1>(t1), std::forward<Tuples>(ts)...);
}


namespace detail
{


template<class T, class Function>
struct tuple_reduce_functor
{
  T& init;
  Function f;

  template<class Arg>
  TUPLE_UTILITY_ANNOTATION
  void operator()(Arg&& arg)
  {
    init = f(init, std::forward<Arg>(arg));
  }
};


} // end detail


template<class Tuple, class T, class Function>
TUPLE_UTILITY_ANNOTATION
T tuple_reduce(Tuple&& t, T init, Function f)
{
  auto g = detail::tuple_reduce_functor<T,Function>{init, f};
  TUPLE_UTILITY_NAMESPACE::tuple_for_each(g, std::forward<Tuple>(t));
  return g.init;
}


namespace detail
{


template<class T>
struct print_element_and_delimiter
{
  std::ostream &os;
  const T& delimiter;

  template<class U>
  void operator()(const U& arg)
  {
    os << arg << delimiter;
  }
};


template<class Tuple, class T>
typename std::enable_if<
  (std::tuple_size<
    decay_t<Tuple>
  >::value > 0)
>::type
  tuple_print_impl(const Tuple& t, std::ostream& os, const T& delimiter)
{
  static const int n_ = std::tuple_size<decay_t<Tuple>>::value - 1;
  static const int n  = n_ < 0 ? 0 : n_;

  tuple_for_each_n<n>(print_element_and_delimiter<T>{os,delimiter}, t);

  // finally print the last element sans delimiter
  os << TUPLE_UTILITY_NAMESPACE::tuple_last(t);
}


} // end detail


template<class Tuple, class T>
void tuple_print(const Tuple& t, std::ostream& os, const T& delimiter)
{
  detail::tuple_print_impl(t, os, delimiter);
}


template<class Tuple>
void tuple_print(const Tuple& t, std::ostream& os = std::cout)
{
  TUPLE_UTILITY_NAMESPACE::tuple_print(t, os, ", ");
}


namespace detail
{


// terminal case: Tuple1 is exhausted
// if we make it to the end of Tuple1, then the tuples are considered equal
template<size_t i, class Tuple1, class Tuple2>
TUPLE_UTILITY_ANNOTATION
typename std::enable_if<
  (i == std::tuple_size<Tuple1>::value) && (i <= std::tuple_size<Tuple2>::value),
  bool
>::type
  tuple_equal_impl(const Tuple1&, const Tuple2&)
{
  return true;
}


// terminal case: Tuple2 is exhausted but not Tuple1
// if we make it to the end of Tuple2 before Tuple1, then the tuples are considered unequal
template<size_t i, class Tuple1, class Tuple2>
TUPLE_UTILITY_ANNOTATION
typename std::enable_if<
  (i < std::tuple_size<Tuple1>::value) && (i == std::tuple_size<Tuple2>::value),
  bool
>::type
  tuple_equal_impl(const Tuple1&, const Tuple2&)
{
  return false;
}


// recursive case
template<size_t i, class Tuple1, class Tuple2>
TUPLE_UTILITY_ANNOTATION
typename std::enable_if<
  (i < std::tuple_size<Tuple1>::value) && (i < std::tuple_size<Tuple2>::value),
  bool
>::type
  tuple_equal_impl(const Tuple1& t1, const Tuple2& t2)
{
  return (TUPLE_UTILITY_NAMESPACE::get<i>(t1) != TUPLE_UTILITY_NAMESPACE::get<i>(t2)) ? false :
         tuple_equal_impl<i+1>(t1, t2);
}


} // end detail


template<class Tuple1, class Tuple2>
TUPLE_UTILITY_ANNOTATION
bool tuple_equal(const Tuple1& t1, const Tuple2& t2)
{
  return detail::tuple_equal_impl<0>(t1,t2);
}


namespace detail
{


template<size_t i, class Tuple1, class Tuple2>
TUPLE_UTILITY_ANNOTATION
typename std::enable_if<
  std::tuple_size<Tuple2>::value <= i,
  bool
>::type
  tuple_lexicographical_compare_impl(const Tuple1&, const Tuple2&)
{
  return false;
}


template<size_t i, class Tuple1, class Tuple2>
TUPLE_UTILITY_ANNOTATION
typename std::enable_if<
  (std::tuple_size<Tuple1>::value <= i && std::tuple_size<Tuple2>::value > i),
  bool
>::type
  tuple_lexicographical_compare_impl(const Tuple1&, const Tuple2&)
{
  return true;
}


template<size_t i, class Tuple1, class Tuple2>
TUPLE_UTILITY_ANNOTATION
typename std::enable_if<
  (std::tuple_size<Tuple1>::value > i && std::tuple_size<Tuple2>::value > i),
  bool
>::type
  tuple_lexicographical_compare_impl(const Tuple1& t1, const Tuple2& t2)
{
  return (TUPLE_UTILITY_NAMESPACE::get<i>(t1) < TUPLE_UTILITY_NAMESPACE::get<i>(t2)) ? true :
         (TUPLE_UTILITY_NAMESPACE::get<i>(t2) < TUPLE_UTILITY_NAMESPACE::get<i>(t1)) ? false :
         tuple_lexicographical_compare_impl<i+1>(t1,t2);
}


} // end detail


template<class Tuple1, class Tuple2>
TUPLE_UTILITY_ANNOTATION
bool tuple_lexicographical_compare(const Tuple1& t1, const Tuple2& t2)
{
  return detail::tuple_lexicographical_compare_impl<0>(t1,t2);
}


namespace detail
{


template<bool b, class True, class False>
struct lazy_conditional
{
  using type = typename True::type;
};


template<class True, class False>
struct lazy_conditional<false, True, False>
{
  using type = typename False::type;
};


template<class T, class U>
struct propagate_reference
{
  using type = U;
};

template<class T, class U>
struct propagate_reference<T&,U>
{
  using type = typename std::add_lvalue_reference<U>::type;
};

template<class T, class U>
struct propagate_reference<T&&,U>
{
  using type = typename std::add_rvalue_reference<U>::type;
};


template<size_t I, class Tuple1, class... Tuples>
struct tuple_cat_element
{
  static const size_t size1 = std::tuple_size<Tuple1>::value;

  using type = typename lazy_conditional<
    (I < size1),
    std::tuple_element<I,Tuple1>,
    tuple_cat_element<I - size1, Tuples...>
  >::type;
};


template<size_t I, class Tuple1>
struct tuple_cat_element<I,Tuple1> : std::tuple_element<I,Tuple1> {};


template<class T, class U>
struct propagate_reference<const T&, U>
{
  using type = const U&;
};


template<size_t i, class TupleReference>
struct tuple_get_result
{
  using type = typename propagate_reference<
    TupleReference,
    typename std::tuple_element<
      i,
      typename std::decay<TupleReference>::type
    >::type
  >::type;
};



template<size_t I, class TupleReference1, class... TupleReferences>
struct tuple_cat_get_result
{
  static_assert(std::is_reference<TupleReference1>::value, "tuple_cat_get_result's template parameters must be reference types.");

  using tuple1_type = typename std::decay<TupleReference1>::type;
  static const size_t size1 = std::tuple_size<tuple1_type>::value;

  using type = typename lazy_conditional<
    (I < size1),
    tuple_get_result<I,TupleReference1>,
    tuple_cat_get_result<I - size1, TupleReferences...>
  >::type;
};


template<size_t I, class TupleReference1>
struct tuple_cat_get_result<I,TupleReference1> : tuple_get_result<I,TupleReference1> {};


template<size_t I, class Tuple1, class... Tuples>
TUPLE_UTILITY_ANNOTATION
typename tuple_cat_get_result<I,Tuple1&&,Tuples&&...>::type
  tuple_cat_get(Tuple1&& t, Tuples&&... ts);


template<size_t I, class Tuple1, class... Tuples>
TUPLE_UTILITY_ANNOTATION
typename tuple_cat_get_result<I,Tuple1&&,Tuples&&...>::type
  tuple_cat_get_impl(std::false_type, Tuple1&& t, Tuples&&...)
{
  return TUPLE_UTILITY_NAMESPACE::get<I>(std::forward<Tuple1>(t));
}


template<size_t I, class Tuple1, class... Tuples>
TUPLE_UTILITY_ANNOTATION
typename tuple_cat_get_result<I,Tuple1&&,Tuples&&...>::type
  tuple_cat_get_impl(std::true_type, Tuple1&&, Tuples&&... ts)
{
  const size_t J = I - std::tuple_size<typename std::decay<Tuple1>::type>::value;
  return detail::tuple_cat_get<J>(std::forward<Tuples>(ts)...);
}


template<size_t I, class Tuple1, class... Tuples>
TUPLE_UTILITY_ANNOTATION
typename tuple_cat_get_result<I,Tuple1&&,Tuples&&...>::type
  tuple_cat_get(Tuple1&& t, Tuples&&... ts)
{
  auto recurse = typename std::conditional<
    I < std::tuple_size<typename std::decay<Tuple1>::type>::value,
    std::false_type,
    std::true_type
  >::type();

  return detail::tuple_cat_get_impl<I>(recurse, std::forward<Tuple1>(t), std::forward<Tuples>(ts)...);
}


template<size_t... I, class Function, class... Tuples>
TUPLE_UTILITY_ANNOTATION
auto tuple_cat_apply_impl(index_sequence<I...>, Function&& f, Tuples&&... ts)
  -> decltype(
       std::forward<Function>(f)(detail::tuple_cat_get<I>(std::forward<Tuples>(ts)...)...)
     )
{
  return std::forward<Function>(f)(detail::tuple_cat_get<I>(std::forward<Tuples>(ts)...)...);
}


template<size_t Size, size_t... Sizes>
struct sum
  : std::integral_constant<
      size_t,
      Size + sum<Sizes...>::value
    >
{};


template<size_t Size> struct sum<Size> : std::integral_constant<size_t, Size> {};


} // end detail


template<class Function, class... Tuples>
TUPLE_UTILITY_ANNOTATION
auto tuple_cat_apply(Function&& f, Tuples&&... ts)
  -> decltype(
       detail::tuple_cat_apply_impl(
         detail::make_index_sequence<
           detail::sum<
             0u,
             std::tuple_size<typename std::decay<Tuples>::type>::value...
           >::value
         >(),
         std::forward<Function>(f),
         std::forward<Tuples>(ts)...
       )
     )
{
  const size_t N = detail::sum<0u, std::tuple_size<typename std::decay<Tuples>::type>::value...>::value;
  return detail::tuple_cat_apply_impl(detail::make_index_sequence<N>(), std::forward<Function>(f), std::forward<Tuples>(ts)...);
}


namespace detail
{


template<class Function, class Tuple, size_t... I>
TUPLE_UTILITY_ANNOTATION
auto tuple_apply_impl(Function f, Tuple&& t, index_sequence<I...>)
  -> decltype(
       f(TUPLE_UTILITY_NAMESPACE::get<I>(std::forward<Tuple>(t))...)
     )
{
  return f(TUPLE_UTILITY_NAMESPACE::get<I>(std::forward<Tuple>(t))...);
}


} // end detail


template<class Function, class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_apply(Function&& f, Tuple&& t)
  -> decltype(
       tuple_cat_apply(std::forward<Function>(f), std::forward<Tuple>(t))
     )
{
  return TUPLE_UTILITY_NAMESPACE::tuple_cat_apply(std::forward<Function>(f), std::forward<Tuple>(t));
}


template<class... Tuples>
TUPLE_UTILITY_ANNOTATION
auto tuple_zip(Tuples&&... tuples)
  -> decltype(
       TUPLE_UTILITY_NAMESPACE::tuple_map(detail::std_tuple_factory{}, std::forward<Tuples>(tuples)...)
     )
{
  return TUPLE_UTILITY_NAMESPACE::tuple_map(detail::std_tuple_factory{}, std::forward<Tuples>(tuples)...);
}


namespace detail
{


// concatenate two index_sequences
template<class IndexSequence1, class IndexSequence2> struct index_sequence_cat_impl;


template<size_t... Indices1, size_t... Indices2>
struct index_sequence_cat_impl<index_sequence<Indices1...>, index_sequence<Indices2...>>
{
  using type = index_sequence<Indices1..., Indices2...>;
};

template<class IndexSequence1, class IndexSequence2>
using index_sequence_cat = typename index_sequence_cat_impl<IndexSequence1,IndexSequence2>::type;


template<template<size_t> class MetaFunction, class Indices>
struct filter_index_sequence_impl;


// an empty sequence filters to the empty sequence
template<template<size_t> class MetaFunction>
struct filter_index_sequence_impl<MetaFunction, index_sequence<>>
{
  using type = index_sequence<>;
};

template<template<size_t> class MetaFunction, size_t Index0, size_t... Indices>
struct filter_index_sequence_impl<MetaFunction, index_sequence<Index0, Indices...>>
{
  // recurse and filter the rest of the indices
  using rest = typename filter_index_sequence_impl<MetaFunction,index_sequence<Indices...>>::type;

  // concatenate Index0 with rest if Index0 passes the filter
  // else, just return rest
  using type = typename std::conditional<
    MetaFunction<Index0>::value,
    index_sequence_cat<
      index_sequence<Index0>,
      rest
    >,
    rest
  >::type;
};


template<template<size_t> class MetaFunction, class Indices>
using filter_index_sequence = typename filter_index_sequence_impl<MetaFunction,Indices>::type;


template<template<class> class MetaFunction, class Tuple>
struct index_filter
{
  using traits = tuple_traits<Tuple>;

  template<size_t i>
  using filter = MetaFunction<typename traits::template element_type<i>>;
};


template<template<class> class MetaFunction, class Tuple>
using make_filtered_indices_for_tuple =
  filter_index_sequence<
    index_filter<MetaFunction, Tuple>::template filter,
    make_index_sequence<tuple_traits<Tuple>::size>
  >;


} // end detail


// XXX nvcc 7.0 has trouble with this template template parameter
//template<template<class> class MetaFunction, class Tuple, class Function>
//TUPLE_UTILITY_ANNOTATION
//auto tuple_filter_invoke(Tuple&& t, Function f)
//  -> decltype(
//       detail::tuple_apply_impl(
//         f,
//         std::forward<Tuple>(t),
//         detail::make_filtered_indices_for_tuple<MetaFunction, typename std::decay<Tuple>::type>{}
//       )
//     )
//{
//  using filtered_indices = detail::make_filtered_indices_for_tuple<MetaFunction, typename std::decay<Tuple>::type>;
//
//  return detail::tuple_apply_impl(f, std::forward<Tuple>(t), filtered_indices{});
//}
template<template<class> class MetaFunction, class Tuple, class Function, class Indices = detail::make_filtered_indices_for_tuple<MetaFunction, typename std::decay<Tuple>::type>>
TUPLE_UTILITY_ANNOTATION
auto tuple_filter_invoke(Tuple&& t, Function f)
  -> decltype(
       detail::tuple_apply_impl(
         f,
         std::forward<Tuple>(t),
         Indices{}
       )
     )
{
  return detail::tuple_apply_impl(f, std::forward<Tuple>(t), Indices{});
}


template<template<class> class MetaFunction, class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_filter(Tuple&& t)
  -> decltype(
       TUPLE_UTILITY_NAMESPACE::tuple_filter_invoke<MetaFunction>(std::forward<Tuple>(t), detail::std_tuple_factory{})
     )
{
  return TUPLE_UTILITY_NAMESPACE::tuple_filter_invoke<MetaFunction>(std::forward<Tuple>(t), detail::std_tuple_factory{});
}


namespace detail
{


template<size_t, class T>
TUPLE_UTILITY_ANNOTATION
T&& identity(T&& x)
{
  return std::forward<T>(x);
}


template<class T, class Function, size_t... Indices>
TUPLE_UTILITY_ANNOTATION
auto tuple_repeat_invoke_impl(T&& x, Function f, index_sequence<Indices...>)
  -> decltype(
       f(identity<Indices>(x)...)
     )
{
  return f(identity<Indices>(x)...);
}


} // end detail


template<size_t N, class T, class Function>
TUPLE_UTILITY_ANNOTATION
auto tuple_repeat_invoke(T&& x, Function f)
  -> decltype(
       detail::tuple_repeat_invoke_impl(std::forward<T>(x), f, detail::make_index_sequence<N>())
     )
{
  return detail::tuple_repeat_invoke_impl(std::forward<T>(x), f, detail::make_index_sequence<N>());
}


template<size_t N, class T>
TUPLE_UTILITY_ANNOTATION
auto tuple_repeat(const T& x)
  -> decltype(
       tuple_repeat_invoke<N>(x, detail::std_tuple_factory{})
     )
{
  return tuple_repeat_invoke<N>(x, detail::std_tuple_factory{});
}


namespace detail
{


template<size_t... Indices, class Tuple, class Function>
TUPLE_UTILITY_ANNOTATION
auto tuple_gather_invoke_impl(Tuple&& t, Function f)
  -> decltype(
       f(TUPLE_UTILITY_NAMESPACE::get<Indices>(std::forward<Tuple>(t))...)
     )
{
  return f(TUPLE_UTILITY_NAMESPACE::get<Indices>(std::forward<Tuple>(t))...);
}


} // end detail


template<size_t... Indices, class Tuple, class Function>
TUPLE_UTILITY_ANNOTATION
auto tuple_gather_invoke(Tuple&& t, Function f)
  -> decltype(
       detail::tuple_gather_invoke_impl<Indices...>(std::forward<Tuple>(t), f)
     )
{
  return detail::tuple_gather_invoke_impl<Indices...>(std::forward<Tuple>(t), f);
}


template<size_t... Indices, class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_gather(Tuple&& t)
  -> decltype(
       tuple_gather_invoke<Indices...>(std::forward<Tuple>(t), detail::std_tuple_factory{})
     )
{
  return tuple_gather_invoke<Indices...>(std::forward<Tuple>(t), detail::std_tuple_factory{});
}


TUPLE_UTILITY_NAMESPACE_CLOSE_BRACE


#ifdef TUPLE_UTILITY_ANNOTATION_NEEDS_UNDEF
#undef TUPLE_UTILITY_ANNOTATION
#undef TUPLE_UTILITY_ANNOTATION_NEEDS_UNDEF
#endif

#ifdef TUPLE_UTILITY_NAMESPACE_NEEDS_UNDEF
#undef TUPLE_UTILITY_NAMESPACE
#undef TUPLE_UTILITY_NAMESPACE_OPEN_BRACE
#undef TUPLE_UTILITY_NAMESPACE_CLOSE_BRACE
#undef TUPLE_UTILITY_NAMESPACE_NEEDS_UNDEF
#endif

#undef TUPLE_UTILITY_REQUIRES

