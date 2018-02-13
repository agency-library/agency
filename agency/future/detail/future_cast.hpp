#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/future/future_traits/is_future.hpp>
#include <agency/future/future_traits/future_rebind_value.hpp>
#include <agency/future.hpp>

#include <type_traits>

namespace agency
{
namespace detail
{
namespace future_cast_detail
{


// two Futures are of the same "kind" if we can rebind the value of one using the other's value
// and we get the same type as the other
// e.g., std::future<T> and std::future<U> are the same kind of future
template<class Future1, class Future2>
struct is_same_kind_of_future
  : std::is_same<
      Future1,
      future_rebind_value_t<
        Future2,
        future_result_t<Future1>
      >
    >
{};


// this version of future_cast_impl() handles the case
// when we are only casting the value type of a future, and not
// changing the kind of future we are dealing with
// in other words, we are casting from e.g. std::future<T> -> std::future<U>
template<class ToFuture, class FromFuture,
         __AGENCY_REQUIRES(is_same_kind_of_future<ToFuture,FromFuture>::value)
        >
__AGENCY_ANNOTATION
ToFuture future_cast_impl(FromFuture& from_future)
{
  using to_result_type = future_result_t<ToFuture>;

  // we can use future_traits in this case
  return future_traits<FromFuture>::template cast<to_result_type>(from_future);
}


template<class FromFuture, class ToResult>
struct future_cast_functor
{
  mutable FromFuture from_future;

  // this handles the case when from_future is a void future
  // this only makes sense when ToResult is also void
  template<class FromResult = future_result_t<FromFuture>,
           class ToResult1 = ToResult,
           __AGENCY_REQUIRES(std::is_void<FromResult>::value),
           __AGENCY_REQUIRES(std::is_void<ToResult1>::value)
          >
  __AGENCY_ANNOTATION
  void operator()() const
  {
    from_future.wait();
  }

  // this handles the case when from_future is a non-void future
  template<class FromValue = future_result_t<FromFuture>,
           __AGENCY_REQUIRES(!std::is_void<FromResult>::value)
          >
  __AGENCY_ANNOTATION
  ToResult operator()() const
  {
    return static_cast<ToResult>(from_future.get());
  }
};


// this version of future_cast_impl() handles the case
// when we are casting the kind of future and possibly the result type as well
// in other words, we are casting from e.g. std::future<T> -> my_future<U>
__agency_exec_check_disable__
template<class ToFuture, class FromFuture,
         __AGENCY_REQUIRES(!is_same_kind_of_future<ToFuture,FromFuture>::value)
        >
__AGENCY_ANNOTATION
ToFuture future_cast_impl(FromFuture& from_future)
{
  // create a ready void future of the same kind as ToFuture
  auto ready = future_traits<ToFuture>::make_ready();

  using to_result_type = future_result_t<ToFuture>;

  // create a continuation to wait on from_future & cast its result
  return agency::future_traits<decltype(ready)>::then(ready, future_cast_functor<FromFuture, to_result_type>{std::move(from_future)});
}


} // end future_cast_detail


template<class ToFuture, class FromFuture,
         __AGENCY_REQUIRES(is_future<ToFuture>::value && is_future<FromFuture>::value),
         __AGENCY_REQUIRES(std::is_convertible<future_result_t<FromFuture>, future_result_t<ToFuture>>::value)
        >
__AGENCY_ANNOTATION
ToFuture future_cast(FromFuture& from_future)
{
  return future_cast_detail::future_cast_impl<ToFuture>(from_future);
}


} // end detail
} // end agency

