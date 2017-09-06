#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/execution_policy/basic_execution_policy.hpp>
#include <agency/cuda/execution/executor/concurrent_executor.hpp>

namespace agency
{
namespace cuda
{


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


// this overload is called on e.g. con.on(cuda::concurrent_executor)
// XXX this function needs to account for the dimensionality of ConcurrentPolicy's agents
template<class ConcurrentPolicy,
         __AGENCY_REQUIRES(
           agency::detail::policy_is_concurrent<ConcurrentPolicy>::value
         )>
__AGENCY_ANNOTATION
cuda::concurrent_execution_policy replace_executor(const ConcurrentPolicy& policy, const concurrent_executor& exec)
{
  return cuda::concurrent_execution_policy(policy.param(), exec);
}


// this overload is called on e.g. con.on(device(0))
// XXX this function needs to account for the dimensionality of ConcurrentPolicy's agents
template<class ConcurrentPolicy,
         __AGENCY_REQUIRES(
           agency::detail::policy_is_concurrent<ConcurrentPolicy>::value
         )>
__AGENCY_ANNOTATION
cuda::concurrent_execution_policy replace_executor(const ConcurrentPolicy& policy, device_id device)
{
  // create a concurrent_executor from the device_id
  cuda::concurrent_executor exec(device);

  // call a lower-level version of replace_executor()
  return cuda::replace_executor(policy, exec);
}


} // end cuda
} // end agency

