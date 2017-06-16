#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/executor/detail/utility/bulk_sync_execute_with_auto_result_and_without_shared_parameters.hpp>
#include <agency/future.hpp>


namespace agency
{
namespace detail
{


template<class SharedFuture>
class bulk_share_future_functor
{
  private:
    SharedFuture sf_;

  public:
    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    bulk_share_future_functor(const SharedFuture& sf)
      : sf_(sf)
    {}

    __AGENCY_ANNOTATION
    bulk_share_future_functor(const bulk_share_future_functor& other)
      : bulk_share_future_functor(other.sf_)
    {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    ~bulk_share_future_functor()
    {}

    __agency_exec_check_disable__
    template<class Index>
    __AGENCY_ANNOTATION
    SharedFuture operator()(const Index&) const
    {
      return sf_;
    }
};


__agency_exec_check_disable__
template<class E, class Future,
         __AGENCY_REQUIRES(Executor<E>())
        >
__AGENCY_ANNOTATION
executor_container<
  E,
  typename future_traits<Future>::shared_future_type
>
  bulk_share_future(E& exec, executor_shape_t<E> shape, Future& f)
{
  using shared_future_type = typename future_traits<Future>::shared_future_type;

  // explicitly share f once to get things started
  shared_future_type shared_f = future_traits<Future>::share(f);

  // bulk execute a function that returns copies of shared_f
  return bulk_sync_execute_with_auto_result_and_without_shared_parameters(exec, bulk_share_future_functor<shared_future_type>(shared_f), shape);
}


} // end detail
} // end agency

