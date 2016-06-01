#pragma once

#include <utility>
#include <functional>
#include <type_traits>
#include <functional>
#include <memory>
#include <tuple>
#include <initializer_list>

#include <agency/execution_agent.hpp>
#include <agency/executor/sequential_executor.hpp>
#include <agency/executor/concurrent_executor.hpp>
#include <agency/executor/parallel_executor.hpp>
#include <agency/executor/vector_executor.hpp>
#include <agency/executor/scoped_executor.hpp>
#include <agency/detail/execution_policy_traits.hpp>
#include <agency/detail/tuple.hpp>

namespace agency
{


template<class T>
struct is_execution_policy : detail::conjunction<
  detail::has_execution_agent_type<T>,
  detail::has_executor<T>,
  detail::has_param<T>
> {};


// customization point -- allow users to specialize this
// to change the type of execution policy based on the type of an executor
template<class ExecutionPolicy, class Executor>
struct rebind_executor;


template<class ExecutionPolicy, class Executor>
using rebind_executor_t = typename rebind_executor<ExecutionPolicy,Executor>::type;


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
// ExecutionCategory, BulkExecutor, ExecutionAgent = __default_execution_agent<ExecutionAgent>, DerivedExecutionPolicy
template<class ExecutionAgent,
         class BulkExecutor,
         class ExecutionCategory = typename execution_agent_traits<ExecutionAgent>::execution_category,
         class DerivedExecutionPolicy = void>
class basic_execution_policy
{
  public:
    using execution_agent_type = ExecutionAgent;
    using executor_type        = BulkExecutor;
    using execution_category   = ExecutionCategory;

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

    template<class OtherExecutor>
    rebind_executor_t<derived_type,OtherExecutor> on(const OtherExecutor& executor) const
    {
      return rebind_executor_t<derived_type,OtherExecutor>(param(), executor);
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

    // this is the scoped form of operator()
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

    template<class Arg1, class... Args>
    derived_type operator()(std::initializer_list<Arg1> arg1, std::initializer_list<Args>... args) const
    {
      return derived_type{param_type{std::move(arg1), std::move(args)...}, executor()};
    }

  protected:
    param_type param_;

    // executor_ needs to be mutable, because:
    // * the global execution policy objects are constexpr
    // * executor.bulk_add() is a non-const member function
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
      scoped_execution_tag<
        typename ExecutionPolicy1::execution_category,
        typename ExecutionPolicy2::execution_category
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
      scoped_execution_tag<
        typename ExecutionPolicy1::execution_category,
        typename ExecutionPolicy2::execution_category
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
struct rebind_executor
{
  using type = detail::basic_execution_policy<
    typename ExecutionPolicy::execution_agent_type,
    Executor
  >;
};


class sequential_execution_policy : public detail::basic_execution_policy<sequential_agent, sequential_executor, sequential_execution_tag, sequential_execution_policy>
{
  private:
    using super_t = detail::basic_execution_policy<sequential_agent, sequential_executor, sequential_execution_tag, sequential_execution_policy>;

  public:
    using super_t::basic_execution_policy;
};


constexpr sequential_execution_policy seq{};


class concurrent_execution_policy : public detail::basic_execution_policy<concurrent_agent, concurrent_executor, concurrent_execution_tag, concurrent_execution_policy>
{
  private:
    using super_t = detail::basic_execution_policy<concurrent_agent, concurrent_executor, concurrent_execution_tag, concurrent_execution_policy>;

  public:
    using super_t::basic_execution_policy;
};


constexpr concurrent_execution_policy con{};


class parallel_execution_policy : public detail::basic_execution_policy<parallel_agent, parallel_executor, parallel_execution_tag, parallel_execution_policy>
{
  private:
    using super_t = detail::basic_execution_policy<parallel_agent, parallel_executor, parallel_execution_tag, parallel_execution_policy>;

  public:
    using super_t::basic_execution_policy;
};


const parallel_execution_policy par{};


class vector_execution_policy : public detail::basic_execution_policy<vector_agent, vector_executor, vector_execution_tag, vector_execution_policy>
{
  private:
    using super_t = detail::basic_execution_policy<vector_agent, vector_executor, vector_execution_tag, vector_execution_policy>;

  public:
    using super_t::basic_execution_policy;
};


constexpr vector_execution_policy vec{};


} // end agency

