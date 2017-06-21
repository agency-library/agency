/// \file
/// \brief Contains definition of unsequenced_execution_policy.
///

#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/unsequenced_executor.hpp>
#include <agency/execution/execution_agent.hpp>
#include <agency/execution/execution_policy/basic_execution_policy.hpp>

namespace agency
{


/// \brief Encapsulates requirements for creating groups of unsequenced execution agents.
/// \ingroup execution_policies
///
///
/// When used as a control structure parameter, `unsequenced_execution_policy` requires the creation of a group of execution agents which execute without any order.
///
/// The type of execution agent `unsequenced_execution_policy` induces is `unsequenced_agent`, and the type of its associated executor is `unsequenced_executor`.
///
/// \see execution_policies
/// \see basic_execution_policy
/// \see vec
/// \see unsequenced_agent
/// \see unsequenced_executor
/// \see unsequenced_execution_tag
class unsequenced_execution_policy : public basic_execution_policy<unsequenced_agent, unsequenced_executor, unsequenced_execution_policy>
{
  private:
    using super_t = basic_execution_policy<unsequenced_agent, unsequenced_executor, unsequenced_execution_policy>;

  public:
    using super_t::basic_execution_policy;
};


/// \brief The global variable `unseq` is the default `unsequenced_execution_policy`.
/// \ingroup execution_policies
constexpr unsequenced_execution_policy unseq{};


/// \brief Encapsulates requirements for creating two-dimensional groups of unsequenced execution agents.
/// \ingroup execution_policies
///
///
/// When used as a control structure parameter, `unsequenced_execution_policy_2d` requires the creation of a two-dimensional group of execution agents which execute without any order.
///
/// The type of execution agent `unsequenced_execution_policy_2d` induces is `unsequenced_agent_2d`, and the type of its associated executor is `unsequenced_executor`.
///
/// \see execution_policies
/// \see basic_execution_policy
/// \see vec
/// \see unsequenced_agent_2d
/// \see unsequenced_executor
/// \see unsequenced_execution_tag
class unsequenced_execution_policy_2d : public basic_execution_policy<unsequenced_agent_2d, unsequenced_executor, unsequenced_execution_policy_2d>
{
  private:
    using super_t = basic_execution_policy<unsequenced_agent_2d, unsequenced_executor, unsequenced_execution_policy_2d>;

  public:
    using super_t::basic_execution_policy;
};


/// \brief The global variable `unseq2d` is the default `unsequenced_execution_policy_2d`.
/// \ingroup execution_policies
constexpr unsequenced_execution_policy_2d unseq2d{};


} // end agency

