#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/exception_list.hpp>
#include <agency/detail/tuple.hpp>
#include <future>
#include <utility>

namespace agency
{
namespace detail
{


template<class T>
std::future<decay_t<T>> make_ready_future(T&& value)
{
  std::promise<decay_t<T>> p;
  p.set_value(std::forward<T>(value));
  return p.get_future();
}


inline std::future<void> make_ready_future()
{
  std::promise<void> p;
  p.set_value();
  return p.get_future();
}


// XXX when_all is supposed to return a future<vector>
template<typename ForwardIterator>
std::future<void> when_all(ForwardIterator first, ForwardIterator last)
{
  exception_list exceptions = flatten_into_exception_list(first, last);

  std::promise<void> p;

  if(exceptions.size() > 0)
  {
    p.set_exception(std::make_exception_ptr(exceptions));
  }
  else
  {
    p.set_value();
  }

  return p.get_future();
}


template<class T, class Function>
std::future<typename std::result_of<Function(std::future<T>&)>::type>
  then(std::future<T>& fut, std::launch policy, Function&& f)
{
  return std::async(policy, [](std::future<T>&& fut, Function&& f)
  {
    fut.wait();
    return std::forward<Function>(f)(fut);
  },
  std::move(fut),
  std::forward<Function>(f)
  );
}


template<class T, class Function>
std::future<typename std::result_of<Function(std::future<T>&)>::type>
  then(std::future<T>& fut, Function&& f)
{
  return detail::then(fut, std::launch::async | std::launch::deferred, std::forward<Function>(f));
}


// XXX should check for nested ::rebind<T> i guess
template<class Future, class T>
struct rebind_future_value;


template<template<class> class Future, class FromType, class ToType>
struct rebind_future_value<Future<FromType>,ToType>
{
  using type = Future<ToType>;
};


__DEFINE_HAS_NESTED_TYPE(has_value_type, value_type);


template<class Future>
struct future_value
{
  using type = decltype(std::declval<Future>().get());
};


namespace is_future_detail
{


template<class T>
struct has_wait_impl
{
  template<class Future,
           class = decltype(
             std::declval<Future>().wait()
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<T>(0));
};


template<class T>
using has_wait = typename has_wait_impl<T>::type;


template<class T>
struct has_get_impl
{
  template<class Future,
           class = decltype(std::declval<Future>().get())
           >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<T>(0));
};


template<class T>
using has_get = typename has_get_impl<T>::type;


} // end is_future_detail


template<class T>
struct is_future
  : std::integral_constant<
      bool,
      is_future_detail::has_wait<T>::value && is_future_detail::has_get<T>::value
    >
{};


template<class T, template<class> class Future, class Enable = void>
struct is_instance_of_future : std::false_type {};

template<class T, template<class> class Future>
struct is_instance_of_future<T,Future,
  typename std::enable_if<
    is_future<T>::value
  >::type
> : std::is_same<
  T,
  Future<
    typename future_value<T>::type
  >
>
{};


} // end detail


template<class Future>
struct future_traits
{
  using future_type = Future;

  using value_type = typename detail::future_value<future_type>::type;

  template<class U>
  using rebind = typename detail::rebind_future_value<future_type,U>::type;

  __AGENCY_ANNOTATION
  static rebind<void> make_ready()
  {
    return rebind<void>::make_ready();
  }

  template<class T>
  __AGENCY_ANNOTATION
  static rebind<typename std::decay<T>::type> make_ready(T&& value)
  {
    return rebind<typename std::decay<T>::type>::make_ready(std::forward<T>(value));
  }

  private:
  template<class Future1>
  struct has_discard_value
  {
    template<class Future2,
             class = decltype(std::declval<Future2*>->discard_value())
            >
    static std::true_type test(int);

    template<class>
    static std::false_type test(...);

    using type = decltype(test<Future1>);
  };

  static rebind<void> discard_value(future_type& fut, std::true_type)
  {
    return fut.discard_value();
  }

  public:

  __AGENCY_ANNOTATION
  static rebind<void> discard_value(future_type& fut)
  {
    return future_traits::discard_value(fut, typename has_discard_value<future_type>::type());
  }
};


template<class T>
struct future_traits<std::future<T>>
{
  using future_type = std::future<T>;

  using value_type = typename detail::future_value<future_type>::type;

  template<class U>
  using rebind = typename detail::rebind_future_value<future_type,U>::type;

  static rebind<void> make_ready()
  {
    return detail::make_ready_future();
  }

  template<class U>
  static rebind<typename std::decay<U>::type> make_ready(U&& value)
  {
    return detail::make_ready_future(std::forward<U>(value));
  }

  template<class Future,
           class = typename std::enable_if<
             std::is_same<Future,future_type>::value &&
             std::is_empty<typename future_traits<Future>::value_type>::value
           >::type>
  static std::future<void> discard_value(Future& fut)
  {
    return std::move(*reinterpret_cast<std::future<void>*>(&fut));
  }
};


namespace detail
{

template<class Tuple, size_t = std::tuple_size<Tuple>::value>
struct unwrap_small_tuple_result
{
  using type = Tuple;
};

template<class Tuple>
struct unwrap_small_tuple_result<Tuple,0>
{
  using type = void;
};

template<class Tuple>
struct unwrap_small_tuple_result<Tuple,1>
{
  using type = typename std::tuple_element<0,Tuple>::type;
};

template<class Tuple>
using unwrap_small_tuple_result_t = typename unwrap_small_tuple_result<Tuple>::type;


template<class Tuple>
void unwrap_small_tuple(Tuple&&,
                        typename std::enable_if<
                          std::tuple_size<
                            typename std::decay<Tuple>::type
                          >::value == 0
                        >::type* = 0)
{}

template<class Tuple>
unwrap_small_tuple_result_t<typename std::decay<Tuple>::type>
  unwrap_small_tuple(Tuple&& t,
                     typename std::enable_if<
                       std::tuple_size<
                         typename std::decay<Tuple>::type
                       >::value == 1
                     >::type* = 0)
{
  return std::move(std::get<0>(t));
}

template<class Tuple>
unwrap_small_tuple_result_t<typename std::decay<Tuple>::type>
  unwrap_small_tuple(Tuple&& t,
                     typename std::enable_if<
                       (std::tuple_size<
                         typename std::decay<Tuple>::type
                       >::value > 1)
                     >::type* = 0)
{
  return std::move(t);
}


template<class TypeList>
struct make_tuple_for_impl;

template<class... Types>
struct make_tuple_for_impl<type_list<Types...>>
{
  using type = detail::tuple<Types...>;
};

template<class TypeList>
using make_tuple_for = typename make_tuple_for_impl<TypeList>::type;


struct void_value {};


template<class T>
struct is_not_void_value : std::integral_constant<bool, !std::is_same<T,void_value>::value> {};


template<class T>
struct void_to_void_value : std::conditional<std::is_void<T>::value, void_value, T> {};


template<class... Futures>
struct tuple_of_future_values_impl
{
  using value_types = type_list<
    typename future_traits<Futures>::value_type...
  >;

  // map void to void_value
  using mapped_value_types = type_list_map<void_to_void_value, value_types>;

  // create a tuple from the list of types
  using type = make_tuple_for<mapped_value_types>;
};

template<class... Futures>
using tuple_of_future_values = typename tuple_of_future_values_impl<Futures...>::type;


template<class Future,
         class = typename std::enable_if<
           std::is_void<
             typename future_traits<Future>::value_type
           >::value
         >::type
        >
void_value get_value(Future& fut)
{
  fut.get();
  return void_value{};
}


template<class Future,
         class = typename std::enable_if<
           !std::is_void<
             typename future_traits<Future>::value_type
           >::value
         >::type
        >
typename future_traits<Future>::value_type
  get_value(Future& fut)
{
  return fut.get();
}


template<class... Futures>
tuple_of_future_values<Futures...>
  get_tuple_of_future_values(Futures&... futures)
{
  return detail::make_tuple(detail::get_value(futures)...);
}


template<class IndexSequence, class... Futures>
struct when_all_and_select_result_impl;

template<size_t... Indices, class... Futures>
struct when_all_and_select_result_impl<index_sequence<Indices...>,Futures...>
{
  using type = decltype(
    detail::unwrap_small_tuple(
      detail::tuple_filter<is_not_void_value>(
        detail::tuple_gather<Indices...>(
          detail::get_tuple_of_future_values(
            *std::declval<Futures*>()...
          )
        )
      )
    )
  );
};


template<class IndexSequence, class... Futures>
using when_all_and_select_result = typename when_all_and_select_result_impl<IndexSequence,Futures...>::type;


template<class IndexSequence, class TypeList>
struct when_all_execute_and_select_result_impl;

template<class IndexSequence, class... Futures>
struct when_all_execute_and_select_result_impl<IndexSequence, type_list<Futures...>>
{
  using type = when_all_and_select_result<IndexSequence,Futures...>;
};


template<class IndexSequence, class TupleOfFutures>
using when_all_execute_and_select_result = typename when_all_execute_and_select_result_impl<IndexSequence, tuple_elements<TupleOfFutures>>::type;


template<size_t... Indices>
struct when_all_execute_and_select_functor
{
  template<class Function, class... Futures>
  __AGENCY_ANNOTATION
  when_all_and_select_result<
    index_sequence<Indices...>, typename std::decay<Futures>::type...
  >
    operator()(Function f, Futures&&... futures)
  {
    auto tuple_of_future_values = detail::get_tuple_of_future_values(futures...);

    // create a view of the non-void values
    auto tuple_of_future_values_view = detail::tuple_filter_view<is_not_void_value>(tuple_of_future_values);

    // invoke f on the non-void values
    detail::tuple_apply(f, tuple_of_future_values_view);

    return detail::unwrap_small_tuple(
      detail::tuple_filter<is_not_void_value>(
        detail::tuple_gather<Indices...>(
          std::move(tuple_of_future_values)
        )
      )
    );
  }
};


template<size_t... Indices, class Function, class... Futures>
std::future<
  detail::when_all_and_select_result<
    detail::index_sequence<Indices...>, typename std::decay<Futures>::type...
  >
>
  when_all_execute_and_select_impl(index_sequence<Indices...>, Function f, Futures&&... futures)
{
  return std::async(std::launch::deferred | std::launch::async, detail::when_all_execute_and_select_functor<Indices...>(), f, std::move(futures)...);
} // end when_all_execute_and_select_impl()


template<size_t... SelectedIndices, size_t... TupleIndices, class TupleOfFutures, class Function>
std::future<
  detail::when_all_execute_and_select_result<
    detail::index_sequence<SelectedIndices...>,
    typename std::decay<TupleOfFutures>::type
  >
>
  when_all_execute_and_select(index_sequence<SelectedIndices...> indices, index_sequence<TupleIndices...>, TupleOfFutures&& futures, Function f)
{
  return detail::when_all_execute_and_select_impl(indices, f, std::get<TupleIndices>(std::forward<TupleOfFutures>(futures))...);
} // end when_all_execute_and_select()


template<class... Futures>
using when_all_result = when_all_and_select_result<index_sequence_for<Futures...>,Futures...>;


struct swallower
{
  template<class... Args>
  void operator()(Args&&...){}
};


} // end detail


template<size_t... Indices, class TupleOfFutures, class Function>
std::future<
  detail::when_all_execute_and_select_result<
    detail::index_sequence<Indices...>,
    typename std::decay<TupleOfFutures>::type
  >
>
  when_all_execute_and_select(TupleOfFutures&& futures, Function f)
{
  return detail::when_all_execute_and_select(
    detail::index_sequence<Indices...>(),
    detail::make_tuple_indices(futures),
    std::forward<TupleOfFutures>(futures),
    f
  );
} // end when_all_execute_and_select()


template<size_t... Indices, class... Futures>
std::future<
  detail::when_all_and_select_result<
    detail::index_sequence<Indices...>, typename std::decay<Futures>::type...
  >
>
  when_all_and_select(Futures&&... futures)
{
  return agency::when_all_execute_and_select<Indices...>(std::make_tuple(std::move(futures)...), detail::swallower());
} // end when_all()


namespace detail
{


template<size_t... Indices, class... Futures>
std::future<
  detail::when_all_result<typename std::decay<Futures>::type...>
>
  when_all_impl(index_sequence<Indices...>, Futures&&... futures)
{
  return agency::when_all_and_select<Indices...>(std::forward<Futures>(futures)...);
} // end when_all()


} // end detail


template<class... Futures>
std::future<
  detail::when_all_result<typename std::decay<Futures>::type...>
>
  when_all(Futures&&... futures)
{
  return detail::when_all_impl(detail::make_index_sequence<sizeof...(futures)>(), std::forward<Futures>(futures)...);
} // end when_all()


} // end agency

