#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/execution/execution_policy/concurrent_execution_policy.hpp>
#include <agency/cuda/execution/execution_policy/concurrent_grid_execution_policy.hpp>
#include <agency/cuda/execution/execution_policy/grid_execution_policy.hpp>
#include <agency/cuda/execution/execution_policy/parallel_execution_policy.hpp>


namespace agency
{
namespace cuda
{
namespace experimental
{


template<size_t group_size, size_t grain_size = 1, size_t pool_size = agency::experimental::default_pool_size(group_size)>
class static_concurrent_execution_policy : public agency::experimental::detail::basic_static_execution_policy<
  cuda::concurrent_execution_policy,
  group_size,
  grain_size,
  agency::experimental::static_concurrent_agent<group_size, grain_size, pool_size>
>
{
  private:
    using super_t = agency::experimental::detail::basic_static_execution_policy<
      cuda::concurrent_execution_policy,
      group_size,
      grain_size,
      agency::experimental::static_concurrent_agent<group_size, grain_size, pool_size>
    >;

  public:
    using super_t::super_t;
};


// XXX consider making this a variable template upon c++17
template<size_t group_size, size_t grain_size = 1, size_t pool_size = agency::experimental::default_pool_size(group_size)>
__AGENCY_ANNOTATION
static_concurrent_execution_policy<group_size, grain_size, pool_size> static_con()
{
  return static_concurrent_execution_policy<group_size, grain_size, pool_size>();
}


// XXX consider making this a global object like the other execution policies
template<size_t block_size, size_t grain_size = 1, size_t heap_size = 0>
auto static_grid(int num_blocks) ->
  decltype(agency::cuda::par(num_blocks, agency::cuda::experimental::static_concurrent_execution_policy<block_size, grain_size, heap_size>()))
{
  return agency::cuda::par(num_blocks, agency::cuda::experimental::static_concurrent_execution_policy<block_size, grain_size, heap_size>());
}

// XXX consider making this a unique type instead of an alias
template<size_t block_size, size_t grain_size = 1, size_t heap_size = 0>
using static_grid_agent = agency::parallel_group<agency::experimental::static_concurrent_agent<block_size, grain_size, heap_size>>;


} // end experimental
} // end cuda
} // end agency

