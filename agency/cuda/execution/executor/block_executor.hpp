#pragma once

#include <agency/cuda/execution/executor/grid_executor.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/detail/tuple.hpp>
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

  // this is the form of operator() for then_execute() with a non-void future and a shared parameter
  template<class Arg1, class Arg2>
  __device__
  auto operator()(grid_executor::index_type idx, Arg1& past_arg, agency::detail::unit, Arg2& inner_shared_param) const ->
    decltype(agency::detail::invoke(f_, agency::detail::get<1>(idx), past_arg, inner_shared_param))
  {
    return agency::detail::invoke(f_, agency::detail::get<1>(idx), past_arg, inner_shared_param);
  }

  // this is the form of operator() for then_execute() with a void future and a shared parameter
  template<class Arg>
  __device__
  auto operator()(grid_executor::index_type idx, agency::detail::unit, Arg& inner_shared_param) const ->
    decltype(agency::detail::invoke(f_, agency::detail::get<1>(idx), inner_shared_param))
  {
    return agency::detail::invoke(f_, agency::detail::get<1>(idx), inner_shared_param);
  }
};


// this container exposes a public interface that makes it look
// like a container that a grid_executor can produce values into (i.e., it is two-dimensional)
// internally, it discards the first dimension of the shape & index and forwards the second
// dimension of each to the actual Container
template<class Container>
class block_executor_helper_container : public Container
{
  private:
    using super_t = Container;

  public:
    __host__ __device__
    block_executor_helper_container()
      : super_t()
    {}

    __host__ __device__
    block_executor_helper_container(Container&& super)
      : super_t(std::move(super))
    {}

    __host__ __device__
    auto operator[](grid_executor::index_type idx) ->
      decltype(this->operator[](agency::detail::get<1>(idx)))
    {
      return super_t::operator[](agency::detail::get<1>(idx));
    }
};


template<class Factory>
struct block_executor_helper_container_factory
{
  Factory factory_;

  using factory_arg_type = typename std::tuple_element<1,grid_executor::shape_type>::type;
  using block_executor_container_type = agency::detail::result_of_t<Factory(factory_arg_type)>;

  __host__ __device__
  block_executor_helper_container<block_executor_container_type> operator()(grid_executor::shape_type shape)
  {
    return block_executor_helper_container<block_executor_container_type>(factory_(agency::detail::get<1>(shape)));
  }
};



// XXX eliminate the stuff above this once we scrub use of old executor_traits
template<class Function>
struct new_block_executor_helper_functor
{
  mutable Function f_;

  // this is the form of operator() for bulk_then_execute() with a non-void predecessor future
  template<class Predecessor, class Result, class InnerSharedArg>
  __device__
  void operator()(grid_executor::index_type idx, Predecessor& predecessor, Result& result, agency::detail::unit, InnerSharedArg& inner_shared_arg) const
  {
    agency::detail::invoke(f_, agency::detail::get<1>(idx), predecessor, result, inner_shared_arg);
  }

  // this is the form of operator() for bulk_then_execute() with a void predecessor future
  template<class Result, class InnerSharedArg>
  __device__
  void operator()(grid_executor::index_type idx, Result& result, agency::detail::unit, InnerSharedArg& inner_shared_arg) const
  {
    agency::detail::invoke(f_, agency::detail::get<1>(idx), result, inner_shared_arg);
  }
};


} // end detail


class block_executor : private grid_executor
{
  private:
    using super_t = grid_executor;

  public:
    using execution_category = concurrent_execution_tag;

    using shape_type = std::tuple_element<1, executor_shape_t<super_t>>::type;
    using index_type = std::tuple_element<1, executor_index_t<super_t>>::type;

    template<class T>
    using future = typename super_t::template future<T>;

    template<class T>
    using allocator = typename super_t::template allocator<T>;

    using super_t::super_t;
    using super_t::make_ready_future;
    using super_t::device;

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
      bulk_then_execute(Function f, shape_type shape, async_future<T>& predecessor, ResultFactory result_factory, SharedFactory shared_factory)
    {
      // wrap f with a functor which accepts indices which grid_executor produces
      auto wrapped_f = detail::new_block_executor_helper_functor<Function>{f};

      // call grid_executor's .bulk_then_execute()
      return super_t::bulk_then_execute(wrapped_f, super_t::shape_type{1,shape}, predecessor, result_factory, agency::detail::unit_factory(), shared_factory);
    }
};


} // end cuda
} // end agency

