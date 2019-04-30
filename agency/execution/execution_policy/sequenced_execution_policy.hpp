/// \file
/// \brief Contains definition of sequenced_execution_policy.
///

#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/sequenced_executor.hpp>
#include <agency/execution/execution_agent.hpp>
#include <agency/execution/execution_policy/basic_execution_policy.hpp>

namespace agency
{


/// \brief Encapsulates requirements for creating groups of sequenced execution agents.
/// \ingroup execution_policies
///
///
/// When used as a control structure parameter, `sequenced_execution_policy` requires the creation of a group of execution agents which execute in sequence.
/// Agents in such a group execute on the thread which invokes the control structure. However, if the executor of a `sequenced_execution_policy` is replaced
/// with another sequenced executor of a different type, the agents may execute in sequence on another thread or threads, depending on the type of that executor.
///
/// The order in which sequenced execution agents execute is given by the lexicographical order of their indices.
///
/// The type of execution agent `sequenced_execution_policy` induces is `sequenced_agent`, and the type of its associated executor is `sequenced_executor`.
///
/// \see execution_policies
/// \see basic_execution_policy
/// \see seq
/// \see sequenced_agent
/// \see sequenced_executor
/// \see sequenced_execution_tag
class sequenced_execution_policy : public basic_execution_policy<sequenced_agent, sequenced_executor, sequenced_execution_policy>
{
  private:
    using super_t = basic_execution_policy<sequenced_agent, sequenced_executor, sequenced_execution_policy>;

  public:
    using super_t::basic_execution_policy;
};


/// \brief The global variable `seq` is the default `sequenced_execution_policy`.
/// \ingroup execution_policies
constexpr sequenced_execution_policy seq{};


/// \brief Encapsulates requirements for creating two-dimensional groups of sequenced execution agents.
/// \ingroup execution_policies
///
///
/// When used as a control structure parameter, `sequenced_execution_policy_2d` requires the creation of a two-dimensional group of execution agents which execute in sequence.
/// Agents in such a group execute on the thread which invokes the control structure. However, if the executor of a `sequenced_execution_policy_2d` is replaced
/// with another sequenced executor of a different type, the agents may execute in sequence on another thread or threads, depending on the type of that executor.
///
/// The order in which sequenced execution agents execute is given by the lexicographical order of their indices.
///
/// The type of execution agent `sequenced_execution_policy_2d` induces is `sequenced_agent_2d`, and the type of its associated executor is `sequenced_executor`.
///
/// \see execution_policies
/// \see basic_execution_policy
/// \see seq
/// \see sequenced_agent_2d
/// \see sequenced_executor
/// \see sequenced_execution_tag
class sequenced_execution_policy_2d : public basic_execution_policy<sequenced_agent_2d, sequenced_executor, sequenced_execution_policy_2d>
{
  private:
    using super_t = basic_execution_policy<sequenced_agent_2d, sequenced_executor, sequenced_execution_policy_2d>;

  public:
    using super_t::basic_execution_policy;
};


/// \brief The global variable `seq2d` is the default `sequenced_execution_policy_2d`.
/// \ingroup execution_policies
constexpr sequenced_execution_policy_2d seq2d{};


/// \brief The function template `seqnd` creates an n-dimensional sequenced
///        execution policy that induces execution agents over the given domain.
/// \ingroup execution_policies
template<class Index>
__AGENCY_ANNOTATION
basic_execution_policy<
  agency::detail::basic_execution_agent<agency::bulk_guarantee_t::sequenced_t,Index>,
  agency::sequenced_executor
>
  seqnd(const agency::lattice<Index>& domain)
{
  using policy_type = agency::basic_execution_policy<
    agency::detail::basic_execution_agent<agency::bulk_guarantee_t::sequenced_t,Index>,
    agency::sequenced_executor
  >;

  typename policy_type::param_type param(domain);
  return policy_type{param};
}


/// \brief The function template `seqnd` creates an n-dimensional sequenced
///         execution policy that creates agent groups of the given shape.
/// \ingroup execution_policies
template<class Shape>
__AGENCY_ANNOTATION
basic_execution_policy<
  agency::detail::basic_execution_agent<agency::bulk_guarantee_t::sequenced_t,Shape>,
  agency::sequenced_executor
>
  seqnd(const Shape& shape)
{
  return agency::seqnd(agency::make_lattice(shape));
}


} // end agency

