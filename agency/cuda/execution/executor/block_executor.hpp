#pragma once

#include <agency/cuda/execution/executor/grid_executor.hpp>
#include <agency/cuda/detail/concurrency/block_barrier.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/tuple.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/detail/type_traits.hpp>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class Function>
struct block_executor_helper_functor
{
  mutable Function f_;

  // this is the form of operator() for bulk_then_execute() with a non-void predecessor future
  template<class Predecessor, class Result, class InnerSharedArg>
  __device__
  void operator()(grid_executor::index_type idx, Predecessor& predecessor, Result& result, agency::detail::unit, InnerSharedArg& inner_shared_arg) const
  {
    agency::detail::invoke(f_, agency::get<1>(idx), predecessor, result, inner_shared_arg);
  }

  // this is the form of operator() for bulk_then_execute() with a void predecessor future
  template<class Result, class InnerSharedArg>
  __device__
  void operator()(grid_executor::index_type idx, Result& result, agency::detail::unit, InnerSharedArg& inner_shared_arg) const
  {
    agency::detail::invoke(f_, agency::get<1>(idx), result, inner_shared_arg);
  }
};


} // end detail


class block_executor : private grid_executor
{
  private:
    using super_t = grid_executor;

  public:
    using shape_type = std::tuple_element<1, executor_shape_t<super_t>>::type;
    using index_type = std::tuple_element<1, executor_index_t<super_t>>::type;

    template<class T>
    using future = typename super_t::template future<T>;

    template<class T>
    using allocator = typename super_t::template allocator<T>;

    using barrier_type = detail::block_barrier;

    using super_t::super_t;
    using super_t::make_ready_future;
    using super_t::device;


    __host__ __device__
    constexpr static bulk_guarantee_t::concurrent_t query(const bulk_guarantee_t&)
    {
      return bulk_guarantee_t::concurrent_t();
    }

    __host__ __device__
    shape_type max_shape_dimensions() const
    {
      return super_t::max_shape_dimensions()[1];
    }

    template<class Function, class T, class ResultFactory, class SharedFactory,
             class = agency::detail::result_of_continuation_t<
               Function,
               index_type,
               async_future<T>,
               agency::detail::result_of_t<ResultFactory()>&,
               agency::detail::result_of_t<SharedFactory()>&
             >
            >
    async_future<agency::detail::result_of_t<ResultFactory()>>
      bulk_then_execute(Function f, shape_type shape, async_future<T>& predecessor, ResultFactory result_factory, SharedFactory shared_factory) const
    {
      // wrap f with a functor which accepts indices which grid_executor produces
      auto wrapped_f = detail::block_executor_helper_functor<Function>{f};

      // call grid_executor's .bulk_then_execute()
      return super_t::bulk_then_execute(wrapped_f, super_t::shape_type{1,shape}, predecessor, result_factory, agency::detail::unit_factory(), shared_factory);
    }
};


} // end cuda
} // end agency

