#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/execution_agent.hpp>
#include <agency/detail/type_traits.hpp>

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


template<class ExecutionPolicy, class T, class Enable = void>
struct execution_policy_future_impl {};

template<class ExecutionPolicy, class T>
struct execution_policy_future_impl<ExecutionPolicy,T,typename std::enable_if<has_executor<ExecutionPolicy>::value>::type>
{
  using type = executor_future_t<execution_policy_executor_t<ExecutionPolicy>, T>;
};


} // end detail


template<class ExecutionPolicy, class T>
struct execution_policy_future
{
  using type = typename detail::execution_policy_future_impl<
    detail::decay_t<ExecutionPolicy>,
    T
  >::type;
};


template<class ExecutionPolicy, class T>
using execution_policy_future_t = typename execution_policy_future<ExecutionPolicy,T>::type;


} // end agency

