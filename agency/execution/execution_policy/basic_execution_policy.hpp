/// \file
/// \brief Contains definition of basic_execution_policy.
///

#pragma once

#include <agency/detail/config.hpp>
#include <agency/tuple.hpp>
#include <agency/execution/execution_agent.hpp>
#include <agency/execution/execution_policy/execution_policy_traits.hpp>
#include <agency/execution/execution_policy/replace_executor.hpp>
#include <agency/execution/executor/associated_executor.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/executor/scoped_executor.hpp>
#include <agency/execution/executor/properties/bulk_guarantee.hpp>

#include <utility>
#include <tuple>
#include <type_traits>
#include <initializer_list>

namespace agency
{
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


} // end detail


/// \brief The basic type from which all of Agency's execution policies derive their common functionality.
/// \ingroup execution_policies
///
///
/// basic_execution_policy defines the essential functionality which all of Agency's execution policies share in common.
/// Because all of Agency's execution policy types publicly inherit from basic_execution_policy, the documentation for
/// their common, public functionality is collected here.
///
/// basic_execution_policy may also be used to define custom execution policy types by instantiating basic_execution_policy
/// with an execution agent type and an executor type. Either of these types may be user-defined.
///
/// \tparam ExecutionAgent The type of execution agent created by the basic_execution_policy.
/// \tparam Executor The type of executor associated with the basic_execution_policy.
/// \tparam DerivedExecutionPolicy The name of the execution policy deriving from this basic_execution_policy.
///         `void` indicates that no execution policy will be derived from this basic_execution_policy.
template<class ExecutionAgent,
         class Executor,
         class DerivedExecutionPolicy = void>
class basic_execution_policy
{
  public:
    // validate that it makes sense to execute the agent's requirements using the executor's guarantees
    static_assert(detail::is_weaker_guarantee_than<
                    typename execution_agent_traits<ExecutionAgent>::execution_requirement,
                    decltype(bulk_guarantee_t::static_query<Executor>())
                  >::value,
                  "basic_execution_policy: ExecutionAgent's forward progress requirements cannot be satisfied by Executor's guarantees."
    );

    /// \brief The type of execution agent associated with this basic_execution_policy.
    using execution_agent_type = ExecutionAgent;

    /// \brief The type of executor associated with this basic_execution_policy.
    using executor_type        = Executor;

  private:
    using derived_type         = typename std::conditional<
      std::is_same<DerivedExecutionPolicy,void>::value,
      basic_execution_policy,
      DerivedExecutionPolicy
    >::type;

    derived_type& derived()
    {
      return static_cast<derived_type&>(*this);
    }

    const derived_type& derived() const
    {
      return static_cast<const derived_type&>(*this);
    }

  public:
    /// \brief The type of this execution policy's parameterization.
    using param_type           = typename execution_agent_traits<execution_agent_type>::param_type;

    /// \brief The default constructor default constructs this execution policy's associated executor and parameterization.
    __agency_exec_check_disable__
    basic_execution_policy() = default;

    /// \brief This constructor constructs a new basic_execution_policy given a parameterization and executor.
    /// \param param The parameterization of this basic_execution_policy.
    /// \param executor The executor to associate with this basic_execution_policy.
    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    basic_execution_policy(const param_type& param, const executor_type& executor)
      : executor_(executor),
        param_(param)
    {}

    /// \brief This constructor constructs a new basic_execution_policy given a parameterization.
    /// \param param The parameterization of this basic_execution_policy.
    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    basic_execution_policy(const param_type& param)
      : basic_execution_policy(param, executor_type{})
    {}

    /// \brief Returns this execution policy's parameterization.
    __AGENCY_ANNOTATION
    const param_type& param() const
    {
      return param_;
    }

    /// \brief Returns this execution policy's associated executor.
    __AGENCY_ANNOTATION
    executor_type& executor() const
    {
      return executor_;
    }
    
    
    /// \brief Returns a copy of this execution policy's executor with the copy's executor replaced with another.
    __agency_exec_check_disable__
    template<class ReplacementExecutor,
             __AGENCY_REQUIRES(
               is_executor<ReplacementExecutor>::value
             )>
    __AGENCY_ANNOTATION
    friend basic_execution_policy<
      ExecutionAgent,
      ReplacementExecutor
    >
      replace_executor(const basic_execution_policy& policy, const ReplacementExecutor& ex)
    {
      using policy_requirement = typename execution_agent_traits<ExecutionAgent>::execution_requirement;
      using executor_guarantee = decltype(bulk_guarantee_t::static_query<ReplacementExecutor>());
    
      static_assert(detail::is_weaker_guarantee_than<policy_requirement, executor_guarantee>::value, "agency::replace_executor(): Execution policy's forward progress requirements cannot be satisfied by executor's guarantees.");
    
      using result_type = basic_execution_policy<
        ExecutionAgent,
        ReplacementExecutor
      >;
    
      return result_type(policy.param(), ex);
    }

    /// \brief Replaces this execution policy's executor with another.
    ///
    ///
    /// on() returns a new execution policy identical to `*this` but
    /// whose associated executor has been replaced by another executor.
    ///
    /// For example, we can require an otherwise parallel task to execute sequentially
    /// in the current thread by executing the task on a sequenced_executor:
    ///
    /// ~~~~{.cpp}
    /// agency::sequenced_executor seq_exec;
    ///
    /// // require the parallel_agents induced by par to execute sequentially on seq_exec
    /// agency::bulk_invoke(agency::par(10).on(seq_exec), [](agency::parallel_agent& self)
    /// {
    ///   std::cout << self.index() << std::endl;
    /// });
    ///
    /// // the integers [0,10) are printed in sequence
    /// ~~~~
    ///
    /// Note that using on() does not change the type of execution agent object created by the policy;
    /// it only changes the underlying physical execution of these agents. The relative
    /// forward progress characteristics of the execution agents required by the execution policy
    /// and the forward progress guarantees must make sense; the forward progress guarantees made by
    /// the executor may not weaken the requirements of the policy. A program that attempts to do this
    /// is ill-formed and will not compile. In this example's case, because agency::sequenced_executor
    /// makes a stronger guarantee (sequenced execution) than does agency::par (parallel execution),
    /// the program is well-formed.
    /// 
    ///
    /// \param exec The other executor to associate with the returned execution policy.
    /// \return An execution policy equivalent to `*this` but whose associated executor is a copy of `exec`.
    ///         The type of the result is an execution policy type `Policy` with the following characteristics:
    ///           * `Policy::execution_agent_type` is `execution_agent_type`,
    ///           * `Policy::param_type` is `param_type`
    ///           * `Policy::executor_type` is `OtherExecutor`.
    /// \note The given executor's forward progress guarantees must not be weaker than this
    ///       execution policy's forward progress requirements.
    /// \note on() is sugar for the expression `agency::replace_executor(static_cast<const DerivedType>(*this), exec)`.
    /// \see replace_executor
    __agency_exec_check_disable__
    template<class OtherExecutor,
             __AGENCY_REQUIRES(is_executor<OtherExecutor>::value)
            >
    __AGENCY_ANNOTATION
    auto on(const OtherExecutor& exec) const ->
      decltype(agency::replace_executor(this->derived(), exec))
    {
      return agency::replace_executor(derived(), exec);
    }

    // XXX probably want to require that agency::associated_executor() is well-formed
    __agency_exec_check_disable__
    template<class T,
             __AGENCY_REQUIRES(!is_executor<detail::decay_t<T>>::value)
            >
    __AGENCY_ANNOTATION
    auto on(T&& has_associated_executor) const ->
      decltype(this->on(agency::associated_executor(std::forward<T>(has_associated_executor))))
    {
      return this->on(agency::associated_executor(std::forward<T>(has_associated_executor)));
    }

    /// \brief Reparameterizes this execution policy.
    ///
    ///
    /// `operator()` returns a new execution policy identical to `*this` but whose
    /// parameterization is constructed from the given arguments.
    ///
    /// \param arg1 The first argument to forward to `param_type`'s constructor.
    /// \param args The rest of the arguments to forward to `param_type`'s constructor.
    /// \return An execution policy equivalent to `*this` but whose parameterization has been constructed from the given arguments.
    ///         The type of the result is:
    ///           * `DerivedExecutionPolicy`, when `DerivedExecutionPolicy` is not `void`
    ///           * `basic_execution_policy<ExecutionAgent,Executor,void>`, otherwise.
    ///
    // this is the flat form of operator()
    // XXX consider introducing .reparamterize() that makes it clearer exactly what is going on
    template<class Arg1, class... Args>
    __AGENCY_ANNOTATION
    #ifndef DOXYGEN_SHOULD_SKIP_THIS
    typename std::enable_if<
      detail::is_flat_call<param_type, Arg1, Args...>::value,
      derived_type
    >::type
    #else
    see_below
    #endif
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
      detail::decay_t<detail::last_type<Arg1,Args...>>
    >
      scope(Arg1&& arg1, Args&&... args) const
    {
      // wrap the args in a tuple so we can manipulate them easier
      auto arg_tuple = agency::forward_as_tuple(std::forward<Arg1>(arg1), std::forward<Args>(args)...);

      // get the arguments to the outer execution policy
      auto outer_args = detail::tuple_drop_last(arg_tuple);

      // create the outer execution policy
      auto outer = agency::apply(*this, outer_args);

      // get the inner execution policy
      auto inner = detail::tuple_last(arg_tuple);

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
        detail::decay_t<detail::last_type<Arg1,Args...>>
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
    // executor_ needs to be mutable, because:
    // * the global execution policy objects are constexpr
    // * executor's member functions are not const
    mutable executor_type executor_;

    param_type param_;
};


namespace detail
{


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
} // end agency

