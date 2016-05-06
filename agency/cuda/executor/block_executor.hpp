#pragma once

#include <agency/cuda/executor/grid_executor.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/functional.hpp>
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
    decltype(agency::invoke(f_, agency::detail::get<1>(idx), past_arg, inner_shared_param))
  {
    return agency::invoke(f_, agency::detail::get<1>(idx), past_arg, inner_shared_param);
  }

  // this is the form of operator() for then_execute() with a void future and a shared parameter
  template<class Arg>
  __device__
  auto operator()(grid_executor::index_type idx, agency::detail::unit, Arg& inner_shared_param) const ->
    decltype(agency::invoke(f_, agency::detail::get<1>(idx), inner_shared_param))
  {
    return agency::invoke(f_, agency::detail::get<1>(idx), inner_shared_param);
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


} // end detail


class block_executor : private grid_executor
{
  private:
    using super_t = grid_executor;
    using super_traits = executor_traits<super_t>;
    using traits = executor_traits<block_executor>;

  public:
    using execution_category = concurrent_execution_tag;

    using shape_type = std::tuple_element<1, super_traits::shape_type>::type;
    using index_type = std::tuple_element<1, super_traits::index_type>::type;

    template<class T>
    using future = super_traits::future<T>;

    template<class T>
    using container = detail::array<T, shape_type>;

    future<void> make_ready_future()
    {
      return super_traits::template make_ready_future<void>(*this);
    }

    using super_t::super_t;
    using super_t::device;

    template<class Function>
    __host__ __device__
    shape_type max_shape(Function f) const
    {
      return super_t::max_shape(f).y;
    }

    template<class Function, class Factory1, class T, class Factory2,
             class = agency::detail::result_of_continuation_t<
               Function,
               index_type,
               async_future<T>,
               agency::detail::result_of_factory_t<Factory2>&
             >
            >
    async_future<agency::detail::result_of_t<Factory1(shape_type)>>
      then_execute(Function f, Factory1 result_factory, shape_type shape, async_future<T>& fut, Factory2 shared_factory)
    {
      // wrap f with a functor which accepts indices which grid_executor produces
      auto wrapped_f = detail::block_executor_helper_functor<Function>{f};

      // wrap result_factory with a factory which creates a container with indices that grid_executor produces
      auto wrapped_result_factory = detail::block_executor_helper_container_factory<Factory1>{result_factory};

      return super_traits::then_execute(*this, wrapped_f, wrapped_result_factory, super_t::shape_type{1,shape}, fut, agency::detail::unit_factory(), shared_factory);
    }
};


} // end cuda
} // end agency

