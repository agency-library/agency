#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/has_member.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/executor_traits.hpp>

namespace agency
{
namespace detail
{


template<class T>
using executor_member_t = typename T::executor_type;

template<class T>
using executor_member_function_t = decay_t<decltype(std::declval<T*>()->executor())>;

template<class ExecutionPolicy>
struct execution_policy_executor
{
  // to detect an execution policy's executor_type,
  // first look for a member type named executor_type,
  // if it does not exist, look for a member function named .executor()
  using type = detected_or_t<
    executor_member_function_t<ExecutionPolicy>,
    executor_member_t, ExecutionPolicy
  >;
};

template<class ExecutionPolicy>
using execution_policy_executor_t = typename execution_policy_executor<ExecutionPolicy>::type;

// XXX nvcc can't correctly compile this implementation of has_executor_member_function in all cases
//// detects whether T::executor() exists
//template<class T>
//using has_executor_member_function = is_detected<executor_member_function_t, T>;


template<class T>
struct has_executor_member_function_impl
{
  template<
    class T2,
    class = decltype(std::declval<T2*>()->executor())
  >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<T>(0));
};

template<class T>
using has_executor_member_function = typename has_executor_member_function_impl<T>::type;

// XXX nvcc can't correctly compile this implementation in all cases
// detects whether T::executor_type exists
//template<class T>
//using has_executor_member_type = is_detected<executor_member_t, T>;

__DEFINE_HAS_MEMBER_TYPE(has_executor_member_type, executor_type);

// detects whether T::executor() or T::executor_type or exist
template<class T>
using has_executor = disjunction<
  has_executor_member_function<T>,
  has_executor_member_type<T>
>;



template<class ExecutionPolicy, class T, class Enable = void>
struct policy_future {};

template<class ExecutionPolicy, class T>
struct policy_future<ExecutionPolicy,T,typename std::enable_if<has_executor<ExecutionPolicy>::value>::type>
{
  using type = executor_future_t<execution_policy_executor_t<ExecutionPolicy>, T>;
};

template<class ExecutionPolicy, class T>
using policy_future_t = typename policy_future<ExecutionPolicy,T>::type;



// XXX nvcc can't correctly compile this implementation in all cases
//template<class T>
//using execution_agent_type_member_t = typename T::execution_agent_type;
//
//template<class T>
//struct execution_policy_agent
//{
//  using type = detected_t<execution_agent_type_member_t, T>;
//};


__DEFINE_HAS_MEMBER_TYPE(has_execution_agent_type, execution_agent_type);

template<class ExecutionPolicy, class Enable = void>
struct execution_policy_agent {};

template<class ExecutionPolicy>
struct execution_policy_agent<ExecutionPolicy,typename std::enable_if<has_execution_agent_type<ExecutionPolicy>::value>::type>
{
  using type = typename ExecutionPolicy::execution_agent_type;
};

template<class T>
using execution_policy_agent_t = typename execution_policy_agent<T>::type;

// XXX nvcc can't correctly compile this implementation in all cases
//template<class T>
//using has_execution_agent_type = is_detected<execution_policy_agent_t, T>;


// returns T's ::execution_agent_type or nonesuch if T has no ::execution_agent_type
template<class T, class Default>
using execution_policy_agent_or_t = lazy_conditional_t<
  has_execution_agent_type<T>::value,
  execution_policy_agent<T>,
  identity<Default>
>;



// XXX nvcc can't correctly compile this implementation of execution_policy_param in all cases
template<class T>
using param_member_t = typename T::param_type;

template<class T>
using param_member_function_t = decay_t<decltype(std::declval<T*>()->param())>;

template<class ExecutionPolicy>
struct execution_policy_param
{
  // to detect an execution policy's param_type,
  // first look for a member type named param_type,
  // if it does not exist, look for a member function named .param()
  using type = detected_or_t<
    param_member_function_t<ExecutionPolicy>,
    param_member_t, ExecutionPolicy
  >;
};

template<class ExecutionPolicy>
using execution_policy_param_t = typename execution_policy_param<ExecutionPolicy>::type;

// XXX nvcc can't correctly compile this implementation of has_param_member_type in all cases
//// detects whether T::param() exists
//template<class T>
//using has_param_member_function = is_detected<param_member_function_t, T>;

template<class T>
struct has_param_member_function_impl
{
  template<
    class T2,
    class = decltype(std::declval<T2*>()->param())
  >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<T>(0));
};

template<class T>
using has_param_member_function = typename has_param_member_function_impl<T>::type;

// XXX nvcc can't correctly compile this implementation of has_param_member_type in all cases
// detects whether T::param_type exists
//template<class T>
//using has_param_member_type = is_detected<param_member_t, T>;

__DEFINE_HAS_MEMBER_TYPE(has_param_member_type, param_type);

// detects whether T::param() or T::param_type or exist
template<class T>
using has_param = disjunction<
  has_param_member_function<T>,
  has_param_member_type<T>
>;


template<class ExecutionPolicy>
struct execution_policy_execution_depth
  : executor_execution_depth<
      execution_policy_executor_t<ExecutionPolicy>
    >
{};


} // end detail
} // end agency

