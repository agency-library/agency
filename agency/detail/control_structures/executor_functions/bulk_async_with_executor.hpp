#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/tuple.hpp>
#include <agency/execution/executor/detail/utility/bulk_twoway_execute_with_void_result.hpp>
#include <agency/execution/executor/detail/utility/bulk_twoway_execute_with_collected_result.hpp>
#include <agency/execution/executor/customization_points/future_cast.hpp>
#include <agency/detail/control_structures/executor_functions/bind_agent_local_parameters.hpp>
#include <agency/detail/control_structures/executor_functions/unpack_shared_parameters_from_executor_and_invoke.hpp>
#include <agency/detail/control_structures/executor_functions/bulk_invoke_with_executor.hpp>
#include <agency/detail/control_structures/executor_functions/result_factory.hpp>
#include <agency/detail/control_structures/scope_result.hpp>
#include <agency/detail/control_structures/decay_parameter.hpp>
#include <agency/detail/type_traits.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{


// this overload handles the general case where the user function returns a normal result
template<class E, class Function, class ResultFactory, class Tuple, size_t... TupleIndices>
__AGENCY_ANNOTATION
executor_future_t<E, result_of_t<ResultFactory()>>
  bulk_async_with_executor_impl(E& exec,
                                Function f,
                                ResultFactory result_factory,
                                executor_shape_t<E> shape,
                                Tuple&& shared_factory_tuple,
                                detail::index_sequence<TupleIndices...>)
{
  return detail::bulk_twoway_execute_with_collected_result(exec, f, shape, result_factory, agency::get<TupleIndices>(std::forward<Tuple>(shared_factory_tuple))...);
}

// this overload handles the special case where the user function returns a scope_result
// the reason we need this case cannot be handled by the overload above is because, unlike the above case,
// there is an intermediate future which must be converted to the right type of result fututre 
template<class E, class Function, size_t scope, class T, class Tuple, size_t... TupleIndices>
__AGENCY_ANNOTATION
executor_future_t<E, typename detail::scope_result_container<scope,T,E>::result_type>
  bulk_async_with_executor_impl(E& exec,
                                Function f,
                                construct<detail::scope_result_container<scope,T,E>, executor_shape_t<E>> result_factory,
                                executor_shape_t<E> shape,
                                Tuple&& shared_factory_tuple,
                                detail::index_sequence<TupleIndices...>)
{
  auto intermediate_future = detail::bulk_twoway_execute_with_collected_result(exec, f, shape, result_factory, agency::get<TupleIndices>(std::forward<Tuple>(shared_factory_tuple))...);

  using result_type = typename detail::scope_result_container<scope,T,E>::result_type;

  // cast the intermediate_future to result_type
  return agency::future_cast<result_type>(exec, intermediate_future);
}

// this overload handles the special case where the user function returns void
template<class E, class Function, class Tuple, size_t... TupleIndices>
__AGENCY_ANNOTATION
executor_future_t<E,void>
  bulk_async_with_executor_impl(E& exec,
                                Function f,
                                void_factory,
                                executor_shape_t<E> shape,
                                Tuple&& factory_tuple,
                                detail::index_sequence<TupleIndices...>)
{
  return detail::bulk_twoway_execute_with_void_result(exec, f, shape, agency::get<TupleIndices>(std::forward<Tuple>(factory_tuple))...);
}


// computes the result type of bulk_async_with_executor
template<class Executor, class Function, class... Args>
struct bulk_async_with_executor_result
{
  using type = executor_future_t<
    Executor, bulk_invoke_with_executor_result_t<Executor,Function,Args...>
  >;
};

template<class Executor, class Function, class... Args>
using bulk_async_with_executor_result_t = typename bulk_async_with_executor_result<Executor,Function,Args...>::type;


template<class Executor, class Function, class... Args>
__AGENCY_ANNOTATION
bulk_async_with_executor_result_t<Executor, Function, Args...>
  bulk_async_with_executor(Executor& exec, executor_shape_t<Executor> shape, Function f, Args&&... args)
{
  // the _1 is for the executor idx parameter, which is the first parameter passed to f
  auto g = detail::bind_agent_local_parameters_workaround_nvbug1754712(std::integral_constant<size_t,1>(), f, detail::placeholders::_1, std::forward<Args>(args)...);

  // make a tuple of the shared args
  auto shared_arg_tuple = detail::forward_shared_parameters_as_tuple(std::forward<Args>(args)...);

  // package up the shared parameters for the executor
  const size_t execution_depth = executor_execution_depth<Executor>::value;

  // create a tuple of factories to use for shared parameters for the executor
  auto factory_tuple = agency::detail::make_shared_parameter_factory_tuple<execution_depth>(shared_arg_tuple);

  // unpack shared parameters we receive from the executor
  auto h = detail::make_unpack_shared_parameters_from_executor_and_invoke(g);

  // compute the type of f's result
  using result_of_f = result_of_t<Function(executor_index_t<Executor>,decay_parameter_t<Args>...)>;

  // based on the type of f's result, make a factory that will create the appropriate type of container to store f's results
  auto result_factory = detail::make_result_factory<result_of_f>(exec, shape);

  return detail::bulk_async_with_executor_impl(exec, h, result_factory, shape, factory_tuple, detail::make_index_sequence<execution_depth>());
}


} // end detail
} // end agency

