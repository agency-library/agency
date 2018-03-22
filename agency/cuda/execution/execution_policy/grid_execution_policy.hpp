#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/execution_policy/basic_execution_policy.hpp>
#include <agency/cuda/execution/execution_policy/parallel_execution_policy.hpp>
#include <agency/cuda/execution/execution_policy/concurrent_execution_policy.hpp>
#include <agency/cuda/execution/executor/grid_executor.hpp>
#include <agency/cuda/execution/executor/multidevice_executor.hpp>
#include <agency/cuda/execution/executor/scoped_executor.hpp>
#include <agency/cuda/device.hpp>


namespace agency
{
namespace cuda
{


// XXX consider making this a global object like the other execution policies
inline auto grid(size_t num_blocks, size_t num_threads) ->
  decltype(
    par(num_blocks, con(num_threads))
  )
{
  return par(num_blocks, con(num_threads));
};


// XXX consider making this a unique type instead of an alias
using grid_agent = parallel_group<concurrent_agent>;


// XXX need to figure out how to make this par(con) select grid_executor_2d automatically
// XXX consider making this a global object like the other execution policies
inline auto grid(size2 grid_dim, size2 block_dim) ->
  decltype(
      par2d(grid_dim, con2d(block_dim)).on(grid_executor_2d())
  )
{
  return par2d(grid_dim, con2d(block_dim)).on(grid_executor_2d());
}

// XXX consider making this a unique type instead of an alias
using grid_agent_2d = parallel_group_2d<concurrent_agent_2d>;


} // end cuda
} // end agency

