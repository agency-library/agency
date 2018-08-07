#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/execution_agent/detail/basic_execution_agent.hpp>
#include <agency/execution/execution_policy/execution_policy_traits.hpp>
#include <agency/execution/executor/sequenced_executor.hpp>

namespace agency
{
namespace detail
{


// simple_sequenced_policy is a simplified version of sequenced_execution_policy.
// The reason it exists is to avoid circular dependencies created between some fancy
// executor types and basic_execution_policy.
//
// Unlike sequenced_execution_policy, simple_sequenced_policy does not inherit from
// basic_execution_policy, and so does not inherit those circular dependencies.
//
// The functionality from sequenced_execution_policy missing from simple_sequenced_policy is .on() and operator().
// Fortunately, the envisioned use cases for simple_sequenced_policy do not require that functionality.
template<class Index = size_t>
class simple_sequenced_policy
{
  public:
    using execution_agent_type = basic_execution_agent<bulk_guarantee_t::sequenced_t, Index>;
    using executor_type = sequenced_executor;
    using param_type = typename execution_agent_traits<execution_agent_type>::param_type;

    __agency_exec_check_disable__
    simple_sequenced_policy() = default;

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    simple_sequenced_policy(const param_type& param, const executor_type& executor = executor_type{})
      : param_(param),
        executor_(executor)
    {}

    __AGENCY_ANNOTATION
    const param_type& param() const
    {
      return param_;
    }

    __AGENCY_ANNOTATION
    executor_type& executor() const
    {
      return executor_;
    }

    template<class... Args>
    simple_sequenced_policy operator()(const Args&... args) const
    {
      return simple_sequenced_policy(param_type(args...));
    }

  private:
    param_type param_;
    mutable executor_type executor_;
};


} // end detail
} // end agency

