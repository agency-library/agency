#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/is_call_possible.hpp>
#include <agency/execution/execution_policy/execution_policy_traits.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{


template<class ExecutionPolicy, class Function, class... Args>
struct is_bulk_call_possible_via_execution_policy_impl
{
  template<class ExecutionPolicy1, class Function1, class... Args1,
           class = typename std::enable_if<
             has_execution_agent_type<ExecutionPolicy1>::value
           >::type,
           class = typename enable_if_call_possible<
             void, Function1, execution_policy_agent_t<ExecutionPolicy1>&, decay_parameter_t<Args1>...
           >::type
          >
  static std::true_type test(int);

  template<class...>
  static std::false_type test(...);

  using type = decltype(test<ExecutionPolicy,Function,Args...>(0));
};

template<class ExecutionPolicy, class Function, class... Args>
using is_bulk_call_possible_via_execution_policy = typename is_bulk_call_possible_via_execution_policy_impl<ExecutionPolicy,Function,Args...>::type;


} // end detail
} // end agency

