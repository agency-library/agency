#pragma once

#include <agency/executor_traits.hpp>

namespace agency
{
namespace detail
{


__DEFINE_HAS_NESTED_TYPE(has_executor_type, executor_type);

template<class ExecutionPolicy, class T, class Enable = void>
struct policy_future {};

// gets the type of future bulk_async(policy) returns
template<class ExecutionPolicy, class T>
struct policy_future<ExecutionPolicy,T,typename std::enable_if<has_executor_type<ExecutionPolicy>::value>::type>
{
  using type = executor_future_t<typename ExecutionPolicy::executor_type, T>;
};

template<class ExecutionPolicy, class T>
using policy_future_t = typename policy_future<ExecutionPolicy,T>::type;


__DEFINE_HAS_NESTED_TYPE(has_execution_agent_type, execution_agent_type);

template<class ExecutionPolicy, class Enable = void>
struct execution_policy_agent {};

template<class ExecutionPolicy>
struct execution_policy_agent<ExecutionPolicy,typename std::enable_if<has_execution_agent_type<ExecutionPolicy>::value>::type>
{
  using type = typename ExecutionPolicy::execution_agent_type;
};


template<class ExecutionPolicy>
using execution_policy_agent_t = typename execution_policy_agent<ExecutionPolicy>::type;




} // end detail
} // end agency

