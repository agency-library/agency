#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/execution/executor/detail/basic_grid_executor.hpp>
#include <agency/coordinate.hpp>
#include <agency/detail/tuple.hpp>


namespace agency
{
namespace cuda
{


class grid_executor : public detail::basic_grid_executor<agency::parallel_execution_tag, agency::uint2>
{
  public:
    using detail::basic_grid_executor<agency::parallel_execution_tag, agency::uint2>::basic_grid_executor;

    __host__ __device__
    shape_type unit_shape() const
    {
      return shape_type{detail::number_of_multiprocessors(device()), 256};
    }

    // XXX does any part of Agency actually use this function? maybe we should just get rid of it
    //     in favor of something that works more like the overload below
    __host__ __device__
    shape_type max_shape_dimensions() const
    {
      // XXX it's not clear that this is correct
      return shape_type{detail::maximum_grid_size_x(device()), 256};
    }

    // this function maximizes each of shape's non-zero elements
    // XXX probably needs a better name
    template<class Function, class T, class ResultFactory, class OuterFactory, class InnerFactory>
    __host__ __device__
    shape_type max_shape_dimensions(Function f, shape_type shape, async_future<T>& predecessor, ResultFactory result_factory, OuterFactory outer_factory, InnerFactory inner_factory) const
    {
      unsigned int outer_size = agency::detail::get<0>(shape);
      unsigned int inner_size = agency::detail::get<1>(shape);

      if(inner_size == 0)
      {
        inner_size = this->max_inner_size(f, predecessor, result_factory, outer_factory, inner_factory);
      }

      if(outer_size == 0)
      {
        outer_size = this->max_parallel_outer_size(f, inner_size, predecessor, result_factory, outer_factory, inner_factory);
      }

      return shape_type{outer_size, inner_size};
    }
};


class grid_executor_2d : public detail::basic_grid_executor<agency::parallel_execution_tag, point<agency::uint2,2>>
{
  public:
    using detail::basic_grid_executor<agency::parallel_execution_tag, point<agency::uint2,2>>::basic_grid_executor;

    // XXX implement unit_shape()

    // XXX implement max_shape_dimensions()
};


} // end cuda
} // end agency

