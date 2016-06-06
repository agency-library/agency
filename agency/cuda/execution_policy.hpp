#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution_policy.hpp>
#include <agency/cuda/executor/grid_executor.hpp>
#include <agency/cuda/executor/parallel_executor.hpp>
#include <agency/cuda/executor/concurrent_executor.hpp>
#include <agency/cuda/executor/scoped_executor.hpp>
#include <agency/detail/tuple.hpp>
#include <type_traits>


namespace agency
{
namespace cuda
{


class parallel_execution_policy : public agency::detail::basic_execution_policy<parallel_agent, cuda::parallel_executor, parallel_execution_tag, parallel_execution_policy>
{
  private:
    using super_t = agency::detail::basic_execution_policy<parallel_agent, cuda::parallel_executor, parallel_execution_tag, parallel_execution_policy>;

    using par2d_t = agency::detail::basic_execution_policy<parallel_agent_2d, cuda::parallel_executor, parallel_execution_tag>;

  public:
    using super_t::basic_execution_policy;

    using super_t::operator();

    // XXX consider whether we really want this functionality a member of parallel_execution_policy
    inline par2d_t operator()(agency::size2 shape) const
    {
      par2d_t par2d;

      return par2d(shape);
    }

    // XXX consider whether we really want this functionality a member of parallel_execution_policy
    template<class ExecutionPolicy>
    agency::detail::scoped_execution_policy<
      par2d_t,
      agency::detail::decay_t<ExecutionPolicy>
    >
      operator()(agency::size2 domain, ExecutionPolicy&& exec) const
    {
      par2d_t par2d;

      return par2d(domain, std::forward<ExecutionPolicy>(exec));
    }
};


const parallel_execution_policy par{};


class concurrent_execution_policy : public agency::detail::basic_execution_policy<concurrent_agent, cuda::concurrent_executor, concurrent_execution_tag, concurrent_execution_policy>
{
  private:
    using super_t = agency::detail::basic_execution_policy<concurrent_agent, cuda::concurrent_executor, concurrent_execution_tag, concurrent_execution_policy>;

    using con2d_t = agency::detail::basic_execution_policy<concurrent_agent_2d, cuda::concurrent_executor, concurrent_execution_tag>;

  public:
    using super_t::basic_execution_policy;

    using super_t::operator();

    // XXX consider whether we really want this functionality a member of concurrent_execution_policy
    inline con2d_t operator()(agency::size2 shape) const
    {
      con2d_t con2d;

      return con2d(shape);
    }

    // XXX consider whether we really want this functionality a member of concurrent_execution_policy
    template<class ExecutionPolicy>
    agency::detail::scoped_execution_policy<
      con2d_t,
      agency::detail::decay_t<ExecutionPolicy>
    >
      operator()(agency::size2 domain, ExecutionPolicy&& exec) const
    {
      con2d_t con2d;

      return con2d(domain, std::forward<ExecutionPolicy>(exec));
    }
};


const concurrent_execution_policy con{};


template<class ExecutionPolicy>
typename std::enable_if<
  std::is_same<
    typename ExecutionPolicy::execution_category,
    concurrent_execution_tag
  >::value,
  cuda::concurrent_execution_policy
>::type
  replace_executor(const ExecutionPolicy& policy, const cuda::concurrent_executor& exec)
{
  return cuda::concurrent_execution_policy(policy.param(), exec);
}


template<class ExecutionPolicy>
typename std::enable_if<
  std::is_same<
    typename ExecutionPolicy::execution_category,
    parallel_execution_tag
  >::value,
  cuda::parallel_execution_policy
>::type
  replace_executor(const ExecutionPolicy& policy, const cuda::parallel_executor& exec)
{
  return cuda::parallel_execution_policy(policy.param(), exec);
}


template<class ExecutionPolicy>
typename std::enable_if<
  std::is_same<
    typename ExecutionPolicy::execution_category,
    parallel_execution_tag
  >::value,
  cuda::parallel_execution_policy
>::type
  replace_executor(const ExecutionPolicy& policy, const cuda::grid_executor& exec)
{
  return cuda::parallel_execution_policy(policy.param(), exec);
}


namespace experimental
{


template<size_t group_size, size_t grain_size = 1>
class static_concurrent_execution_policy : public agency::experimental::detail::basic_static_execution_policy<cuda::concurrent_execution_policy, group_size, grain_size>
{
  private:
    using super_t = agency::experimental::detail::basic_static_execution_policy<cuda::concurrent_execution_policy, group_size, grain_size>;

  public:
    using super_t::super_t;
};


} // end experimental
} // end cuda
} // end agency

