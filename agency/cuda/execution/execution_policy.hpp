#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/execution_policy.hpp>
#include <agency/execution/detail/execution_policy_traits.hpp>
#include <agency/cuda/execution/executor/grid_executor.hpp>
#include <agency/cuda/execution/executor/parallel_executor.hpp>
#include <agency/cuda/execution/executor/concurrent_executor.hpp>
#include <agency/cuda/execution/executor/scoped_executor.hpp>
#include <agency/detail/tuple.hpp>
#include <type_traits>


namespace agency
{
namespace cuda
{


class parallel_execution_policy : public basic_execution_policy<parallel_agent, cuda::parallel_executor, parallel_execution_policy>
{
  private:
    using super_t = basic_execution_policy<parallel_agent, cuda::parallel_executor, parallel_execution_policy>;

  public:
    using super_t::basic_execution_policy;
};


const parallel_execution_policy par{};


class parallel_execution_policy_2d : public basic_execution_policy<parallel_agent_2d, cuda::parallel_executor, parallel_execution_policy_2d>
{
  private:
    using super_t = basic_execution_policy<parallel_agent_2d, cuda::parallel_executor, parallel_execution_policy_2d>;

  public:
    using super_t::basic_execution_policy;
};


const parallel_execution_policy_2d par2d{};


class concurrent_execution_policy : public basic_execution_policy<concurrent_agent, cuda::concurrent_executor, concurrent_execution_policy>
{
  private:
    using super_t = basic_execution_policy<concurrent_agent, cuda::concurrent_executor, concurrent_execution_policy>;

  public:
    using super_t::basic_execution_policy;
};


const concurrent_execution_policy con{};


class concurrent_execution_policy_2d : public basic_execution_policy<concurrent_agent_2d, cuda::concurrent_executor, concurrent_execution_policy_2d>
{
  private:
    using super_t = basic_execution_policy<concurrent_agent_2d, cuda::concurrent_executor, concurrent_execution_policy_2d>;

  public:
    using super_t::basic_execution_policy;
};


const concurrent_execution_policy_2d con2d{};


// XXX this function needs to account for the dimensionality of ExecutionPolicy's agents
template<class ExecutionPolicy>
__AGENCY_ANNOTATION
typename std::enable_if<
  std::is_same<
    agency::detail::execution_policy_execution_category_t<ExecutionPolicy>,
    concurrent_execution_tag
  >::value,
  cuda::concurrent_execution_policy
>::type
  replace_executor(const ExecutionPolicy& policy, const cuda::concurrent_executor& exec)
{
  return cuda::concurrent_execution_policy(policy.param(), exec);
}


// XXX this function needs to account for the dimensionality of ExecutionPolicy's agents
template<class ExecutionPolicy>
__AGENCY_ANNOTATION
typename std::enable_if<
  std::is_same<
    agency::detail::execution_policy_execution_category_t<ExecutionPolicy>,
    parallel_execution_tag
  >::value,
  cuda::parallel_execution_policy
>::type
  replace_executor(const ExecutionPolicy& policy, const cuda::parallel_executor& exec)
{
  return cuda::parallel_execution_policy(policy.param(), exec);
}


// XXX this function needs to account for the dimensionality of ExecutionPolicy's agents
template<class ExecutionPolicy>
__AGENCY_ANNOTATION
typename std::enable_if<
  std::is_same<
    agency::detail::execution_policy_execution_category_t<ExecutionPolicy>,
    parallel_execution_tag
  >::value,
  cuda::parallel_execution_policy
>::type
  replace_executor(const ExecutionPolicy& policy, const cuda::grid_executor& exec)
{
  return cuda::parallel_execution_policy(policy.param(), exec);
}


// XXX consider making this a global object like the other execution policies
auto grid(size_t num_blocks, size_t num_threads) ->
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
auto grid(size2 grid_dim, size2 block_dim) ->
  decltype(
      par2d(grid_dim, con2d(block_dim)).on(grid_executor_2d())
  )
{
  return par2d(grid_dim, con2d(block_dim)).on(grid_executor_2d());
}

// XXX consider making this a unique type instead of an alias
using grid_agent_2d = parallel_group_2d<concurrent_agent_2d>;


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

