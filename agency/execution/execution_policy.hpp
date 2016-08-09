/// \file
/// \brief Contains definitions of built-in execution policies.
///

/// \defgroup execution_policies Execution Policies
/// \ingroup execution
/// \brief Execution policies describe requirements for execution.
///
/// Execution policies describe the execution properties of bulk tasks created by control structures such as `bulk_invoke()`.
/// Such properties include both *how* and *where* execution should occur. Forward progress requirements encapsulated by 
/// execution policies describe the ordering relationships of individual execution agents comprising a bulk task, while the execution policy's
/// associated *executor* governs where those execution agents execute.

#pragma once

#include <utility>
#include <functional>
#include <type_traits>
#include <functional>
#include <memory>
#include <tuple>
#include <initializer_list>

#include <agency/detail/tuple.hpp>
#include <agency/execution/executor.hpp>
#include <agency/execution/execution_agent.hpp>
#include <agency/execution/detail/execution_policy_traits.hpp>

namespace agency
{


template<class T>
struct is_execution_policy : detail::conjunction<
  detail::has_execution_agent_type<T>,
  detail::has_executor<T>,
  detail::has_param<T>
> {};


namespace detail
{


// declare basic_execution_policy for replace_executor()'s signature below 
template<class ExecutionAgent,
         class BulkExecutor,
         class DerivedExecutionPolicy = void>
class basic_execution_policy;


} // end detail


// declare replace_executor() so basic_execution_policy.on() can use it below
template<class ExecutionPolicy, class Executor>
detail::basic_execution_policy<
  typename ExecutionPolicy::execution_agent_type,
  Executor>
replace_executor(const ExecutionPolicy& policy, const Executor& exec);


namespace detail
{


template<class... Types>
struct last_type_impl
{
  typedef typename std::tuple_element<sizeof...(Types) - 1, std::tuple<Types...>>::type type;
};


template<>
struct last_type_impl<>
{
  typedef void type;
};


template<class... Types>
using last_type = typename last_type_impl<Types...>::type;


template<class ParamType, class... Args>
struct is_scoped_call
  : std::integral_constant<
      bool,
      is_execution_policy<last_type<Args...>>::value &&
      is_constructible_from_type_list<
        ParamType,
        type_list_drop_last<
          type_list<Args...>
        >
      >::value
    >
{};


template<class ParamType, class... Args>
struct is_flat_call
  : std::integral_constant<
      bool,
      is_constructible_from_type_list<ParamType, type_list<Args...>>::value
    >
{};


// declare scoped_execution_policy for basic_execution_policy's use below
template<class ExecutionPolicy1, class ExecutionPolicy2>
class scoped_execution_policy;


// XXX we should assert that ExecutionCategory is stronger than the category of ExecutionAgent
// XXX another way to order these parameters would be
// BulkExecutor, ExecutionAgent = __default_execution_agent<ExecutionAgent>, DerivedExecutionPolicy
template<class ExecutionAgent,
         class BulkExecutor,
         class DerivedExecutionPolicy>
class basic_execution_policy
{
  public:
    using execution_agent_type = ExecutionAgent;
    using executor_type        = BulkExecutor;
    using execution_category   = typename execution_agent_traits<ExecutionAgent>::execution_category;

  private:
    using derived_type         = typename std::conditional<
      std::is_same<DerivedExecutionPolicy,void>::value,
      basic_execution_policy,
      DerivedExecutionPolicy
    >::type;

    using agent_traits         = execution_agent_traits<execution_agent_type>;

  public:
    using param_type           = typename agent_traits::param_type;

    basic_execution_policy() = default;

    basic_execution_policy(const param_type& param, const executor_type& executor = executor_type{})
      : param_(param),
        executor_(executor)
    {}

    const param_type& param() const
    {
      return param_;
    }

    executor_type& executor() const
    {
      return executor_;
    }

    // .on() is just sugar for replace_executor(*this, executor())
    template<class OtherExecutor>
    auto on(const OtherExecutor& exec) const ->
      decltype(replace_executor(*this, exec))
    {
      // note the intentional use of ADL to call replace_executor()
      return replace_executor(*this, exec);
    }

    // this is the flat form of operator()
    template<class Arg1, class... Args>
    typename std::enable_if<
      detail::is_flat_call<param_type, Arg1, Args...>::value,
      derived_type
    >::type
      operator()(Arg1&& arg1, Args&&... args) const
    {
      return derived_type{param_type{std::forward<Arg1>(arg1), std::forward<Args>(args)...}, executor()};
    }

    // XXX maybe .scope() should just take OuterPolicy & InnerPolicy?
    //     instead of a bunch of args?
    // XXX seems like scope() should require at least two arguments
    template<class Arg1, class... Args>
    detail::scoped_execution_policy<
      derived_type,
      decay_t<last_type<Arg1,Args...>>
    >
      scope(Arg1&& arg1, Args&&... args) const
    {
      // wrap the args in a tuple so we can manipulate them easier
      auto arg_tuple = detail::forward_as_tuple(std::forward<Arg1>(arg1), std::forward<Args>(args)...);

      // get the arguments to the outer execution policy
      auto outer_args = detail::tuple_drop_last(arg_tuple);

      // create the outer execution policy
      auto outer = detail::tuple_apply(*this, outer_args);

      // get the inner execution policy
      auto inner = __tu::tuple_last(arg_tuple);

      // return the composition of the two policies
      return detail::scoped_execution_policy<derived_type,decltype(inner)>(outer, inner);
    }

    // this is the scoped form of operator()
    // it is just sugar for .scope()
    template<class Arg1, class... Args>
    typename std::enable_if<
      detail::is_scoped_call<param_type, Arg1, Args...>::value,
      detail::scoped_execution_policy<
        derived_type,
        decay_t<last_type<Arg1,Args...>>
      >
    >::type
      operator()(Arg1&& arg1, Args&&... args) const
    {
      return scope(std::forward<Arg1>(arg1), std::forward<Args>(args)...);
    }

    template<class Arg1, class... Args>
    derived_type operator()(std::initializer_list<Arg1> arg1, std::initializer_list<Args>... args) const
    {
      return derived_type{param_type{std::move(arg1), std::move(args)...}, executor()};
    }

  protected:
    param_type param_;

    // executor_ needs to be mutable, because:
    // * the global execution policy objects are constexpr
    // * executor's member functions are not const
    mutable executor_type executor_;
};


template<class ExecutionPolicy1, class ExecutionPolicy2>
class scoped_execution_policy
  : public basic_execution_policy<
      execution_group<
        typename ExecutionPolicy1::execution_agent_type,
        typename ExecutionPolicy2::execution_agent_type
      >,
      scoped_executor<
        typename ExecutionPolicy1::executor_type,
        typename ExecutionPolicy2::executor_type
      >,
      scoped_execution_policy<ExecutionPolicy1,ExecutionPolicy2>
    >
{
  private:
    using super_t = basic_execution_policy<
      execution_group<
        typename ExecutionPolicy1::execution_agent_type,
        typename ExecutionPolicy2::execution_agent_type
      >,
      scoped_executor<
        typename ExecutionPolicy1::executor_type,
        typename ExecutionPolicy2::executor_type
      >,
      scoped_execution_policy<ExecutionPolicy1,ExecutionPolicy2>
    >;


  public:
    using outer_execution_policy_type = ExecutionPolicy1;
    using inner_execution_policy_type = ExecutionPolicy2;
    using typename super_t::execution_agent_type;
    using typename super_t::executor_type;

    scoped_execution_policy(const outer_execution_policy_type& outer,
                            const inner_execution_policy_type& inner)
      : super_t(typename execution_agent_type::param_type(outer.param(), inner.param()),
                executor_type(outer.executor(), inner.executor())),
        outer_(outer),
        inner_(inner)
    {}

    const outer_execution_policy_type& outer() const
    {
      return outer_;
    }

    const inner_execution_policy_type& inner() const
    {
      return inner_;
    }

  private:
    outer_execution_policy_type outer_;
    inner_execution_policy_type inner_;
};


} // end detail


template<class ExecutionPolicy, class Executor>
detail::basic_execution_policy<
  typename ExecutionPolicy::execution_agent_type,
  Executor
>
replace_executor(const ExecutionPolicy& policy, const Executor& exec)
{
  using policy_category = typename ExecutionPolicy::execution_category;
  using executor_category = detail::executor_execution_category_t<Executor>;

  static_assert(detail::is_weaker_than<policy_category, executor_category>::value, "replace_executor(): Execution policy's forward progress requirements cannot be satisfied by executor's guarantees.");

  using result_type = detail::basic_execution_policy<
    typename ExecutionPolicy::execution_agent_type,
    Executor
  >;

  return result_type(policy.param(), exec);
}


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
/// The type of execution agent `sequenced_execution_policy` induces is `sequenced_agent`.
///
/// \see seq
/// \see sequenced_agent
/// \see sequenced_executor
/// \see sequenced_execution_tag
class sequenced_execution_policy : public detail::basic_execution_policy<sequenced_agent, sequenced_executor, sequenced_execution_policy>
{
  private:
    using super_t = detail::basic_execution_policy<sequenced_agent, sequenced_executor, sequenced_execution_policy>;

  public:
    using super_t::basic_execution_policy;
};


/// \brief The global variable `seq` is the default `sequenced_execution_policy`.
/// \ingroup execution_policies
constexpr sequenced_execution_policy seq{};


/// \brief Encapsulates requirements for creating groups of concurrent execution agents.
/// \ingroup execution_policies
///
///
/// When used as a control structure parameter, `concurrent_execution_policy` requires the creation of a group of execution agents which execute concurrently.
/// Agents in such a group are guaranteed to make forward progress.
///
/// The type of execution agent `concurrent_execution_policy` induces is `concurrent_agent`.
///
/// \see con
/// \see concurrent_agent
/// \see concurrent_executor
/// \see concurrent_execution_tag
class concurrent_execution_policy : public detail::basic_execution_policy<concurrent_agent, concurrent_executor, concurrent_execution_policy>
{
  private:
    using super_t = detail::basic_execution_policy<concurrent_agent, concurrent_executor, concurrent_execution_policy>;

  public:
    using super_t::basic_execution_policy;
};


/// \brief The global variable `con` is the default `concurrent_execution_policy`.
/// \ingroup execution_policies
constexpr concurrent_execution_policy con{};


/// \brief Encapsulates requirements for creating groups of parallel execution agents.
/// \ingroup execution_policies
///
///
/// When used as a control structure parameter, `parallel_execution_policy` requires the creation of a group of execution agents which execute in parallel.
/// When agents in such a group execute on separate threads, they have no order. Otherwise, if agents in such a group execute on the same thread,
/// they execute in an unspecified order.
///
/// The type of execution agent `parallel_execution_policy` induces is `parallel_agent`.
///
/// \see par
/// \see parallel_agent
/// \see parallel_executor
/// \see parallel_execution_tag
class parallel_execution_policy : public detail::basic_execution_policy<parallel_agent, parallel_executor, parallel_execution_policy>
{
  private:
    using super_t = detail::basic_execution_policy<parallel_agent, parallel_executor, parallel_execution_policy>;

  public:
    using super_t::basic_execution_policy;
};


/// \brief The global variable `par` is the default `parallel_execution_policy`.
/// \ingroup execution_policies
const parallel_execution_policy par{};


class vector_execution_policy : public detail::basic_execution_policy<vector_agent, vector_executor, vector_execution_policy>
{
  private:
    using super_t = detail::basic_execution_policy<vector_agent, vector_executor, vector_execution_policy>;

  public:
    using super_t::basic_execution_policy;
};


constexpr vector_execution_policy vec{};


namespace experimental
{
namespace detail
{


template<class ExecutionPolicy, std::size_t group_size,
         std::size_t grain_size = 1,
         class ExecutionAgent = basic_static_execution_agent<
           agency::detail::execution_policy_agent_t<ExecutionPolicy>,
           group_size,
           grain_size
         >,
         class Executor       = agency::detail::execution_policy_executor_t<ExecutionPolicy>>
using basic_static_execution_policy = agency::detail::basic_execution_policy<
  ExecutionAgent,
  Executor
>;


} // end detail


template<size_t group_size, size_t grain_size = 1>
class static_sequenced_execution_policy : public detail::basic_static_execution_policy<agency::sequenced_execution_policy, group_size, grain_size>
{
  private:
    using super_t = detail::basic_static_execution_policy<agency::sequenced_execution_policy, group_size, grain_size>;

  public:
    using super_t::super_t;
};


template<size_t group_size, size_t grain_size = 1>
class static_concurrent_execution_policy : public detail::basic_static_execution_policy<
  agency::concurrent_execution_policy,
  group_size,
  grain_size,
  static_concurrent_agent<group_size, grain_size>
>
{
  private:
    using super_t = detail::basic_static_execution_policy<
      agency::concurrent_execution_policy,
      group_size,
      grain_size,
      static_concurrent_agent<group_size, grain_size>
    >;

  public:
    using super_t::super_t;
};


} // end experimental
} // end agency

