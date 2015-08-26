#pragma once

#include <agency/future.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{
namespace future_cast_implementation_strategies
{


template<class FromFuture, class ToFuture>
struct is_future_castable_impl
{
  using from_value_type = typename agency::future_traits<FromFuture>::value_type;
  using to_value_type   = typename agency::future_traits<ToFuture>::value_type;

  using cast_type = decltype(
    agency::future_traits<FromFuture>::template cast<to_value_type>(*std::declval<FromFuture*>())
  );

  using type = std::is_same<cast_type, ToFuture>;
};


template<class FromFuture, class ToFuture>
using is_future_castable = typename is_future_castable_impl<FromFuture,ToFuture>::type;


// strategies for future_cast implementation
struct use_future_cast_member_function {};

struct use_future_traits {};

struct use_then_execute {};


template<class Executor, class T, class Future>
using select_future_cast_implementation = 
  typename std::conditional<
    has_future_cast<Executor,T,Future>::value,
    use_future_cast_member_function,
    typename std::conditional<
      is_future_castable<Future, typename new_executor_traits<Executor>::template future<T>>::value,
      use_future_traits,
      use_then_execute
    >::type
  >::type;


} // end future_cast_implementation_strategies


template<class T, class Executor, class Future>
typename new_executor_traits<Executor>::template future<T>
  future_cast(future_cast_implementation_strategies::use_future_cast_member_function, Executor& ex, Future& fut)
{
  return ex.template future_cast<T>(ex, fut);
} // end future_cast()


template<class T, class Executor, class Future>
typename new_executor_traits<Executor>::template future<T>
  future_cast(future_cast_implementation_strategies::use_future_traits, Executor&, Future& fut)
{
  return agency::future_traits<Future>::template cast<T>(fut);
} // end future_cast()


template<class T>
struct future_cast_then_execute_functor
{
  // cast from U -> T
  template<class U>
  __AGENCY_ANNOTATION
  T operator()(U& arg) const
  {
    return static_cast<T>(std::move(arg));
  }

  // cast from void -> void
  // T would be void in this case
  __AGENCY_ANNOTATION
  T operator()() const
  {
  }
};


template<class T, class Executor, class Future>
typename new_executor_traits<Executor>::template future<T>
  future_cast(future_cast_implementation_strategies::use_then_execute, Executor& ex, Future& fut)
{
  return new_executor_traits<Executor>::then_execute(ex, future_cast_then_execute_functor<T>{}, fut);
} // end future_cast()


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class T, class Future>
typename new_executor_traits<Executor>::template future<T>
  new_executor_traits<Executor>
    ::future_cast(typename new_executor_traits<Executor>::executor_type& ex, Future& fut)
{
  using namespace agency::detail::new_executor_traits_detail::future_cast_implementation_strategies;

  using implementation_strategy = select_future_cast_implementation<
    Executor,
    T,
    Future
  >;

  return detail::new_executor_traits_detail::future_cast<T>(implementation_strategy(), ex, fut);
} // end future_cast()


} // end agency

