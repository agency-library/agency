#pragma once

#include <agency/detail/config.hpp>
#include <agency/executor_traits.hpp>
#include <agency/execution_agent.hpp>
#include <agency/functional.hpp>
#include <agency/detail/bulk_functions/decay_parameter.hpp>
#include <agency/detail/bulk_functions/shared_parameter.hpp>
#include <agency/detail/bulk_functions/bind_agent_local_parameters.hpp>
#include <agency/detail/is_call_possible.hpp>
#include <agency/detail/shape_cast.hpp>
#include <agency/detail/index_cast.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/execution_policy_traits.hpp>
#include <agency/detail/bulk_functions/single_result.hpp>
#include <agency/detail/bulk_functions/result_factory.hpp>
#include <agency/detail/type_traits.hpp>


// XXX this has gotten complicated, so we should reorganize the implementation of bulk_invoke()
//     into headers organized under agency/detail/bulk_invoke/*.hpp
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

// XXX move these to the top of this header
#include <agency/detail/bulk_functions/bulk_invoke.hpp>
#include <agency/detail/bulk_functions/bulk_async.hpp>
#include <agency/detail/bulk_functions/bulk_then.hpp>

