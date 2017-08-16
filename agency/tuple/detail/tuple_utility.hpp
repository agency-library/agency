#pragma once

#include <agency/detail/config.hpp>


// XXX this #include should be eliminated from this file
#define __TUPLE_ANNOTATION __AGENCY_ANNOTATION
#define __TUPLE_NAMESPACE __tu
#include <agency/tuple/detail/tuple_impl.hpp>
#undef __TUPLE_ANNOTATION
#undef __TUPLE_NAMESPACE

// XXX this #include should be eliminated from this file
#define TUPLE_UTILITY_ANNOTATION __AGENCY_ANNOTATION
#define TUPLE_UTILITY_NAMESPACE __tu
#include <agency/tuple/detail/tuple_utility_impl.hpp>
#undef TUPLE_UTILITY_ANNOTATION
#undef TUPLE_UTILITY_NAMESPACE


#include <agency/detail/requires.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/type_list.hpp>
#include <agency/detail/host_device_cast.hpp>
#include <agency/detail/has_member.hpp>
#include <agency/detail/type_traits.hpp>
#include <utility>
#include <type_traits>


namespace agency
{
namespace detail
{


__DEFINE_HAS_MEMBER(has_value, value);


template<class T>
struct is_tuple : has_value<std::tuple_size<T>> {};


// fancy version of std::get which uses tuple_traits and can get() from things which aren't in std::
template<size_t i, class Tuple,
         class = typename std::enable_if<
           is_tuple<typename std::decay<Tuple>::type>::value
         >::type>
__AGENCY_ANNOTATION
auto get(Tuple&& t)
  -> decltype(
       __tu::tuple_traits<typename std::decay<Tuple>::type>::template get<i>(std::forward<Tuple>(t))
     )
{
  return __tu::tuple_traits<typename std::decay<Tuple>::type>::template get<i>(std::forward<Tuple>(t));
}


// get_if returns the ith element of an object when that object is a Tuple-like type
// otherwise, it returns its second parameter
template<size_t i, class Tuple, class T,
         __AGENCY_REQUIRES(
           is_tuple<typename std::decay<Tuple>::type>::value
         )>
__AGENCY_ANNOTATION
auto get_if(Tuple&& t, T&&)
  -> decltype(get<i>(std::forward<Tuple>(t)))
{
  return detail::get<i>(std::forward<Tuple>(t));
}


template<size_t, class NotATuple, class T,
         __AGENCY_REQUIRES(
           !is_tuple<typename std::decay<NotATuple>::type>::value
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


struct forwarder
{
  template<class... Args>
  __AGENCY_ANNOTATION
  auto operator()(Args&&... args)
    -> decltype(
         __tu::forward_as_tuple(std::forward<Args>(args)...)
       )
  {
    return __tu::forward_as_tuple(std::forward<Args>(args)...);
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
         __tu::make_tuple(std::move(args)...)
       )
  {
    return __tu::make_tuple(std::move(args)...);
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
    return __tu::make_tuple(std::forward<Args>(args)...);
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
auto tuple_head_if(T&& t) ->
  decltype(detail::get<0>(std::forward<T>(t)))
{
  return detail::get<0>(std::forward<T>(t));
}


template<class T,
         class = typename std::enable_if<
           !is_tuple<typename std::decay<T>::type>::value
         >::type>
__AGENCY_ANNOTATION
T&& tuple_head_if(T&& t)
{
  return std::forward<T>(t);
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
__tu::tuple<> tuple_tail_if(T&&)
{
  return __tu::tuple<>();
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
__tu::tuple<> tuple_prefix_if(T&&)
{
  return __tu::tuple<>();
}


template<class T,
         class = typename std::enable_if<
           is_tuple<typename std::decay<T>::type>::value
         >::type>
__AGENCY_ANNOTATION
auto tuple_last_if(T&& t) ->
  decltype(__tu::tuple_last(std::forward<T>(t)))
{
  return __tu::tuple_last(std::forward<T>(t));
}


template<class T,
         class = typename std::enable_if<
           !is_tuple<typename std::decay<T>::type>::value
         >::type>
__AGENCY_ANNOTATION
T&& tuple_last_if(T&& t)
{
  return std::forward<T>(t);
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
auto tuple_apply(Function&& f, Tuple&& t)
  -> decltype(
       __tu::tuple_apply(agency::detail::host_device_cast(std::forward<Function>(f)), std::forward<Tuple>(t))
     )
{
  return __tu::tuple_apply(agency::detail::host_device_cast(std::forward<Function>(f)), std::forward<Tuple>(t));
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
       detail::get<0>(std::forward<Tuple>(t))
     )
{
  return detail::get<0>(std::forward<Tuple>(t));
}


// if the argument is a tuple, it unwraps it if it is a single-element tuple,
// otherwise, it returns the tuple
// if the argument is not a tuple, it returns the argument
template<class Tuple,
         class = typename std::enable_if<
           is_tuple<typename std::decay<Tuple>::type>::value
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
           !is_tuple<typename std::decay<T>::type>::value
         >::type>
T&& unwrap_single_element_tuple_if(T&& arg)
{
  return std::forward<T>(arg);
}


template<class TupleReference, class IndexSequence>
struct decay_tuple_impl;


template<class TupleReference, size_t... Indices>
struct decay_tuple_impl<TupleReference, index_sequence<Indices...>>
{
  using tuple_type = typename std::decay<TupleReference>::type;

  using type = __tu::tuple<
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
  using type = __tu::tuple<Types...>;
};

template<class T, size_t size>
using homogeneous_tuple = typename homogeneous_tuple_impl<type_list_repeat<size,T>>:: type;


template<size_t size, class T>
__AGENCY_ANNOTATION
homogeneous_tuple<T,size> make_homogeneous_tuple(const T& val)
{
  return detail::tuple_repeat<size>(val);
}


// this is the inverse operation of tuple_elements
template<class TypeList>
struct tuple_from_type_list;

template<class... Types>
struct tuple_from_type_list<agency::detail::type_list<Types...>>
{
  using type = __tu::tuple<Types...>;
};

template<class TypeList>
using tuple_from_type_list_t = typename tuple_from_type_list<TypeList>::type;


template<class TypeList>
struct tuple_or_single_type_or_void_from_type_list
{
  using type = tuple_from_type_list_t<TypeList>;
};

template<class T>
struct tuple_or_single_type_or_void_from_type_list<agency::detail::type_list<T>>
{
  using type = T;
};

template<>
struct tuple_or_single_type_or_void_from_type_list<agency::detail::type_list<>>
{
  using type = void;
};


template<class TypeList>
using tuple_or_single_type_or_void_from_type_list_t = typename tuple_or_single_type_or_void_from_type_list<TypeList>::type;


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
struct tuple_type_list<Tuple, typename std::enable_if<is_tuple<Tuple>::value>::type>
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
struct is_empty_tuple_impl<Tuple, typename std::enable_if<is_tuple<Tuple>::value>::type>
{
  using type = typename is_empty_tuple_impl_impl<
    typename tuple_type_list<Tuple>::type
  >::type;
};


template<class Tuple>
struct is_empty_tuple : is_empty_tuple_impl<Tuple>::type {};


template<class Tuple>
using tuple_reverse_t = tuple_from_type_list_t<
  type_list_reverse<
    tuple_elements<Tuple>
  >
>;


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


// tuple_rebind takes a Tuple-like type and reinstantiates it with a different list of types
template<class Tuple, class... Types>
struct tuple_rebind;


// we can tuple_rebind a Tuple-like type simply by reinstantiating the template from which it came
template<template<class...> class TupleLike, class... OriginalTypes, class... Types>
struct tuple_rebind<TupleLike<OriginalTypes...>, Types...>
{
  using type = TupleLike<Types...>;
};



// we can tuple_rebind an Array-like type only when the list of Types are all the same type
template<template<class,size_t> class ArrayLike, class OriginalType, size_t n, class Type, class... Types>
struct tuple_rebind<ArrayLike<OriginalType,n>, Type, Types...>
  : std::conditional<
      conjunction<std::is_same<Type,Types>...>::value,  // if all of Types are the same as Type
      ArrayLike<Type, 1 + sizeof...(Types)>,            // then reinstantiate the Array-like template using Type
      std::enable_if<false>                             // otherwise, do not define a member named ::type
    >::type
{};


template<class Tuple, class... Types>
using tuple_rebind_t = typename tuple_rebind<Tuple,Types...>::type;


// a Tuple-like type is rebindable for a list of types if tuple_rebind<Tuple,Types...>::type is detected to exist
// XXX WAR nvbug 1965139
//template<class Tuple, class... Types>
//using is_tuple_rebindable = is_detected<tuple_rebind_t, Tuple, Types...>;
template<class Tuple, class... Types>
struct is_tuple_rebindable : is_detected<tuple_rebind_t, Tuple, Types...> {};


// some types aren't tuple_rebindable given a list of Types
// in such cases, we default to using the given TupleLike template as the result of the rebind
template<class T, template<class...> class TupleLike, class... Types>
using tuple_rebind_if = lazy_conditional<
  is_tuple_rebindable<T,Types...>::value, // if Tuple is rebindable...
  tuple_rebind<T,Types...>,               // then tuple_rebind it
  identity<TupleLike<Types...>>           // otherwise, default to TupleLike<Types...>
>;

template<class T, template<class...> class TupleLike, class... Types>
using tuple_rebind_if_t = typename tuple_rebind_if<T,TupleLike,Types...>::type;


} // end detail
} // end agency

