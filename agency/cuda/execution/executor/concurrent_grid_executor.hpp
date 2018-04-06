#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/execution/executor/detail/basic_grid_executor.hpp>
#include <agency/execution/executor/properties/bulk_guarantee.hpp>
#include <agency/coordinate/point.hpp>


namespace agency
{
namespace cuda
{


class concurrent_grid_executor : public detail::basic_grid_executor<bulk_guarantee_t::concurrent_t, agency::uint2>
{
  private:
    using super_t = detail::basic_grid_executor<bulk_guarantee_t::concurrent_t, agency::uint2>;

  public:
    using super_t::super_t;

    __host__ __device__
    shape_type unit_shape() const
    {
      return shape_type{detail::number_of_multiprocessors(device()), 256};
    }

    using super_t::max_shape_dimensions;

    // XXX does any part of Agency actually use this function? maybe we should just get rid of it
    //     in favor of something that works more like the overload below
    __host__ __device__
    shape_type max_shape_dimensions() const
    {
      // XXX it's not clear that this is correct
      return shape_type{detail::maximum_grid_size_x(device()), detail::maximum_block_size_x(device())};
    }
};


} // end cuda
} // end agency

