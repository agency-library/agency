#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/future.hpp>
#include <agency/execution/executor/detail/execution_functions/then_execute.hpp>
#include <agency/execution/executor/executor_traits/executor_future.hpp>
#include <agency/execution/executor/executor_traits/is_executor.hpp>
#include <utility>
#include <type_traits>


namespace agency
{
namespace detail
{


template<class T, class Executor, class Future>
struct has_future_cast_impl
{
  using expected_future_type = executor_future_t<Executor, T>;

  template<
    class Executor2,
    class Result = decltype(
      std::declval<Executor2&>().template future_cast<T>(std::declval<Future&>())
    ),
    __AGENCY_REQUIRES(std::is_same<expected_future_type,Result>::value)
  >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};


template<class T, class Executor, class Future>
using has_future_cast = typename has_future_cast_impl<T,Executor,Future>::type;


// this overload handles executors which have .future_cast()
__agency_exec_check_disable__
template<class T, class E, class Future,
         __AGENCY_REQUIRES(has_future_cast<T,E,Future>::value)
        >
__AGENCY_ANNOTATION
executor_future_t<E,T> future_cast_impl(const E& exec, Future& fut)
{
  return exec.template future_cast<T>(fut);
}


template<class FromFuture, class ToFuture>
struct is_future_castable_impl
{
  using from_result_type = future_result_t<FromFuture>;
  using to_result_type   = future_result_t<ToFuture>;

  using cast_type = decltype(
    agency::future_traits<FromFuture>::template cast<to_result_type>(std::declval<FromFuture&>())
  );

  using type = std::is_same<cast_type, ToFuture>;
};


template<class FromFuture, class ToFuture>
using is_future_castable = typename is_future_castable_impl<FromFuture,ToFuture>::type;


// this overload handles executors which do not have .future_cast()
// and when future_traits::cast() may be used
template<class T, class E, class Future,
         __AGENCY_REQUIRES(is_executor<E>::value),
         __AGENCY_REQUIRES(!has_future_cast<T,E,Future>::value),
         __AGENCY_REQUIRES(is_future_castable<Future, executor_future_t<E,T>>::value)
        >
__AGENCY_ANNOTATION
executor_future_t<E,T> future_cast_impl(const E&, Future& fut)
{
  return future_traits<Future>::template cast<T>(fut);
}


template<class T>
struct future_cast_functor
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


// this overload handles executors which do not have .future_cast(),
// and when future_traits::cast() may not be used
// in this case, we create a continuation to execute the conversion
template<class T, class E, class Future,
         __AGENCY_REQUIRES(is_executor<E>::value),
         __AGENCY_REQUIRES(!has_future_cast<T,E,Future>::value),
         __AGENCY_REQUIRES(!is_future_castable<Future, executor_future_t<E,T>>::value)
        >
__AGENCY_ANNOTATION
executor_future_t<E,T> future_cast_impl(const E& exec, Future& fut)
{
  return agency::detail::then_execute(exec, future_cast_functor<T>(), fut);
}


} // end detail


__agency_exec_check_disable__
template<class T, class E, class Future>
__AGENCY_ANNOTATION
executor_future_t<E,T> future_cast(const E& exec, Future& fut)
{
  return detail::future_cast_impl<T>(exec, fut);
} // end future_cast()


} // end agency

