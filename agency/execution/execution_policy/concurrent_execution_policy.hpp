/// \file
/// \brief Contains definition of concurrent_execution_policy.
///

#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/concurrent_executor.hpp>
#include <agency/execution/execution_agent.hpp>
#include <agency/execution/execution_policy/basic_execution_policy.hpp>

namespace agency
{


/// \brief Encapsulates requirements for creating groups of concurrent execution agents.
/// \ingroup execution_policies
///
///
/// When used as a control structure parameter, `concurrent_execution_policy` requires the creation of a group of execution agents which execute concurrently.
/// Agents in such a group are guaranteed to make forward progress.
///
/// The type of execution agent `concurrent_execution_policy` induces is `concurrent_agent`, and the type of its associated executor is `concurrent_executor`.
///
/// \see execution_policies
/// \see basic_execution_policy
/// \see con
/// \see concurrent_agent
/// \see concurrent_executor
/// \see concurrent_execution_tag
class concurrent_execution_policy : public basic_execution_policy<concurrent_agent, concurrent_executor, concurrent_execution_policy>
{
  private:
    using super_t = basic_execution_policy<concurrent_agent, concurrent_executor, concurrent_execution_policy>;

  public:
    using super_t::basic_execution_policy;
};


/// \brief The global variable `con` is the default `concurrent_execution_policy`.
/// \ingroup execution_policies
constexpr concurrent_execution_policy con{};


/// \brief Encapsulates requirements for creating two-dimensional groups of concurrent execution agents.
/// \ingroup execution_policies
///
///
/// When used as a control structure parameter, `concurrent_execution_policy_2d` requires the creation of a two-dimensional group of execution agents which execute concurrently.
/// Agents in such a group are guaranteed to make forward progress.
///
/// The type of execution agent `concurrent_execution_policy_2d` induces is `concurrent_agent_2d`, and the type of its associated executor is `concurrent_executor`.
///
/// \see execution_policies
/// \see basic_execution_policy
/// \see con
/// \see concurrent_agent_2d
/// \see concurrent_executor
/// \see concurrent_execution_tag
class concurrent_execution_policy_2d : public basic_execution_policy<concurrent_agent_2d, concurrent_executor, concurrent_execution_policy_2d>
{
  private:
    using super_t = basic_execution_policy<concurrent_agent_2d, concurrent_executor, concurrent_execution_policy_2d>;

  public:
    using super_t::basic_execution_policy;
};


/// \brief The global variable `con2d` is the default `concurrent_execution_policy_2d`.
/// \ingroup execution_policies
constexpr concurrent_execution_policy_2d con2d{};


} // end agency

