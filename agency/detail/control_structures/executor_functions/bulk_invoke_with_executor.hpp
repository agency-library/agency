#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/tuple.hpp>
#include <agency/execution/executor/detail/utility/executor_bulk_result_or_void.hpp>
#include <agency/execution/executor/detail/utility/blocking_bulk_twoway_execute_with_void_result.hpp>
#include <agency/execution/executor/detail/utility/blocking_bulk_twoway_execute_with_collected_result.hpp>
#include <agency/detail/control_structures/executor_functions/bind_agent_local_parameters.hpp>
#include <agency/detail/control_structures/executor_functions/unpack_shared_parameters_from_executor_and_invoke.hpp>
#include <agency/detail/control_structures/executor_functions/result_factory.hpp>
#include <agency/detail/control_structures/scope_result.hpp>
#include <agency/detail/control_structures/decay_parameter.hpp>
#include <agency/detail/type_traits.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{


// this overload handles the general case where the user function returns a non-void result
template<class Executor, class Function, class ResultFactory, class Tuple, size_t... TupleIndices>
__AGENCY_ANNOTATION
result_of_t<ResultFactory()>
  bulk_invoke_with_executor_impl(Executor& exec,
                                 Function f,
                                 ResultFactory result_factory,
                                 executor_shape_t<Executor> shape,
                                 Tuple&& shared_factory_tuple,
                                 detail::index_sequence<TupleIndices...>)
{
  return detail::blocking_bulk_twoway_execute_with_collected_result(exec, f, shape, result_factory, agency::get<TupleIndices>(std::forward<Tuple>(shared_factory_tuple))...);
}

// this overload handles the special case where the user function returns void
template<class Executor, class Function, class Tuple, size_t... TupleIndices>
__AGENCY_ANNOTATION
void bulk_invoke_with_executor_impl(Executor& exec,
                                    Function f,
                                    void_factory,
                                    executor_shape_t<Executor> shape,
                                    Tuple&& factory_tuple,
                                    detail::index_sequence<TupleIndices...>)
{
  return detail::blocking_bulk_twoway_execute_with_void_result(exec, f, shape, agency::get<TupleIndices>(std::forward<Tuple>(factory_tuple))...);
}


// computes the result type of bulk_invoke(executor)
template<class Executor, class Function, class... Args>
struct bulk_invoke_with_executor_result
{
  // first figure out what type the user function returns
  using user_function_result = result_of_t<
    Function(executor_index_t<Executor>, decay_parameter_t<Args>...)
  >;

  // if the user function returns scope_result, then use scope_result_to_bulk_invoke_result to figure out what to return
  // else, the result is whatever executor_bulk_result_or_void<Executor, function_result> thinks it is
  using type = typename lazy_conditional<
    is_scope_result<user_function_result>::value,
    scope_result_to_bulk_invoke_result<user_function_result, Executor>,
    executor_bulk_result_or_void<Executor, user_function_result>
  >::type;
};

template<class Executor, class Function, class... Args>
using bulk_invoke_with_executor_result_t = typename bulk_invoke_with_executor_result<Executor,Function,Args...>::type;


template<class Executor, class Function, class... Args>
__AGENCY_ANNOTATION
bulk_invoke_with_executor_result_t<Executor, Function, Args...>
  bulk_invoke_with_executor(Executor& exec, executor_shape_t<Executor> shape, Function f, Args&&... args)
{
  // the _1 is for the executor idx parameter, which is the first parameter passed to f
  auto g = detail::bind_agent_local_parameters_workaround_nvbug1754712(std::integral_constant<size_t,1>(), f, detail::placeholders::_1, std::forward<Args>(args)...);

  // make a tuple of the shared args
  auto shared_arg_tuple = detail::forward_shared_parameters_as_tuple(std::forward<Args>(args)...);

  // package up the shared parameters for the executor
  const size_t execution_depth = executor_execution_depth<Executor>::value;

  // create a tuple of factories to use for shared parameters for the executor
  auto factory_tuple = detail::make_shared_parameter_factory_tuple<execution_depth>(shared_arg_tuple);

  // unpack shared parameters we receive from the executor
  auto h = detail::make_unpack_shared_parameters_from_executor_and_invoke(g);

  // compute the type of f's result
  using result_of_f = result_of_t<Function(executor_index_t<Executor>,decay_parameter_t<Args>...)>;

  // based on the type of f's result, make a factory that will create the appropriate type of container to store f's results
  auto result_factory = detail::make_result_factory<result_of_f>(exec, shape);

  return detail::bulk_invoke_with_executor_impl(exec, h, result_factory, shape, factory_tuple, detail::make_index_sequence<execution_depth>());
}


} // end detail
} // end agency

