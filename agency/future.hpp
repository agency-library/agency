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


template<class... Futures>
struct tuple_of_future_values_impl
{
  // want to create type_list<future_value_t<Future>, future_value_t<Futures>...>
  // and filter void

  // create a list of the value types
  using value_types = type_list<
    typename future_traits<Futures>::value_type...
  >;

  template<class T>
  struct is_not_void : std::integral_constant<bool, !std::is_void<T>::value> {};

  // get the list of types which are not void
  using non_void_value_types = type_list_filter<is_not_void, value_types>;

  // create a tuple from the list of types
  using type = make_tuple_for<non_void_value_types>;
};


template<class... Futures>
using tuple_of_future_values = typename tuple_of_future_values_impl<Futures...>::type;


template<template<class> class FutureTemplate, class... Futures>
using when_all_result_t = FutureTemplate<
  unwrap_small_tuple_result_t<tuple_of_future_values<Futures...>>
>;


template<class Future,
         class = typename std::enable_if<
           std::is_void<
             typename future_traits<Future>::value_type
           >::value
         >::type
        >
detail::tuple<> wrap_value_in_tuple(Future& fut)
{
  fut.get();
  return detail::make_tuple();
}


template<class Future,
         class = typename std::enable_if<
           !std::is_void<
             typename future_traits<Future>::value_type
           >::value
         >::type
        >
detail::tuple<typename future_traits<Future>::value_type> wrap_value_in_tuple(Future& fut)
{
  return detail::make_tuple(fut.get());
}


inline tuple_of_future_values<> get_tuple_of_future_values()
{
  return tuple_of_future_values<>();
}


template<class Future, class... Futures>
tuple_of_future_values<Future,Futures...>
  get_tuple_of_future_values(Future& future, Futures&... futures)
{
  auto head = wrap_value_in_tuple(future);
  return detail::tuple_cat(head, get_tuple_of_future_values(futures...));
}


struct when_all_functor
{
  template<class... Futures>
  unwrap_small_tuple_result_t<
    tuple_of_future_values<
      typename std::decay<Futures>::type...
    >
  >
    operator()(Futures&&... futures)
  {
    return unwrap_small_tuple(get_tuple_of_future_values(futures...));
  }
};


} // end detail


template<class... Futures>
detail::when_all_result_t<
  std::future,
  typename std::decay<Futures>::type...
>
  when_all(Futures&&... futures)
{
  return std::async(std::launch::deferred | std::launch::async, detail::when_all_functor(), std::move(futures)...);
} // end when_all_result()


} // end agency

