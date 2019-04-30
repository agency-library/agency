/// \file
/// \brief Contains definition of parallel_execution_policy.
///

#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/parallel_executor.hpp>
#include <agency/execution/execution_agent.hpp>
#include <agency/execution/execution_policy/basic_execution_policy.hpp>


namespace agency
{


/// \brief Encapsulates requirements for creating groups of parallel execution agents.
/// \ingroup execution_policies
///
///
/// When used as a control structure parameter, `parallel_execution_policy` requires the creation of a group of execution agents which execute in parallel.
/// When agents in such a group execute on separate threads, they have no order. Otherwise, if agents in such a group execute on the same thread,
/// they execute in an unspecified order.
///
/// The type of execution agent `parallel_execution_policy` induces is `parallel_agent`, and the type of its associated executor is `parallel_executor`.
///
/// \see execution_policies
/// \see basic_execution_policy
/// \see par
/// \see parallel_agent
/// \see parallel_executor
/// \see parallel_execution_tag
class parallel_execution_policy : public basic_execution_policy<parallel_agent, parallel_executor, parallel_execution_policy>
{
  private:
    using super_t = basic_execution_policy<parallel_agent, parallel_executor, parallel_execution_policy>;

  public:
    using super_t::basic_execution_policy;
};


/// \brief The global variable `par` is the default `parallel_execution_policy`.
/// \ingroup execution_policies
const parallel_execution_policy par{};


/// \brief Encapsulates requirements for creating two-dimensional groups of parallel execution agents.
/// \ingroup execution_policies
///
///
/// When used as a control structure parameter, `parallel_execution_policy_2d` requires the creation of a two-dimensional group of execution agents which execute in parallel.
/// When agents in such a group execute on separate threads, they have no order. Otherwise, if agents in such a group execute on the same thread,
/// they execute in an unspecified order.
///
/// The type of execution agent `parallel_execution_policy_2d` induces is `parallel_agent_2d`, and the type of its associated executor is `parallel_executor`.
///
/// \see execution_policies
/// \see basic_execution_policy
/// \see par
/// \see parallel_agent_2d
/// \see parallel_executor
/// \see parallel_execution_tag
class parallel_execution_policy_2d : public basic_execution_policy<parallel_agent_2d, parallel_executor, parallel_execution_policy_2d>
{
  private:
    using super_t = basic_execution_policy<parallel_agent_2d, parallel_executor, parallel_execution_policy_2d>;

  public:
    using super_t::basic_execution_policy;
};


/// \brief The global variable `par2d` is the default `parallel_execution_policy_2d`.
/// \ingroup execution_policies
const parallel_execution_policy_2d par2d{};


/// \brief The function template `parnd` creates an n-dimensional parallel
///        execution policy that induces execution agents over the given domain.
/// \ingroup execution_policies
template<class Index>
__AGENCY_ANNOTATION
basic_execution_policy<
  agency::detail::basic_execution_agent<agency::bulk_guarantee_t::parallel_t,Index>,
  agency::parallel_executor
>
  parnd(const agency::lattice<Index>& domain)
{
  using policy_type = agency::basic_execution_policy<
    agency::detail::basic_execution_agent<agency::bulk_guarantee_t::parallel_t,Index>,
    agency::parallel_executor
  >;

  typename policy_type::param_type param(domain);
  return policy_type{param};
}


/// \brief The function template `parnd` creates an n-dimensional parallel
///         execution policy that creates agent groups of the given shape.
/// \ingroup execution_policies
template<class Shape>
__AGENCY_ANNOTATION
basic_execution_policy<
  agency::detail::basic_execution_agent<agency::bulk_guarantee_t::parallel_t,Shape>,
  agency::parallel_executor
>
  parnd(const Shape& shape)
{
  return agency::parnd(agency::make_lattice(shape));
}


} // end agency

