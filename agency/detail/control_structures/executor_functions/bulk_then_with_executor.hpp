#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/tuple.hpp>
#include <agency/execution/executor/customization_points/future_cast.hpp>
#include <agency/execution/executor/detail/utility/bulk_then_execute_with_void_result.hpp>
#include <agency/execution/executor/detail/utility/bulk_then_execute_with_collected_result.hpp>
#include <agency/detail/control_structures/executor_functions/bind_agent_local_parameters.hpp>
#include <agency/detail/control_structures/executor_functions/bulk_async_with_executor.hpp>
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
template<class E, class Function, class ResultFactory, class Future, class Tuple, size_t... TupleIndices>
__AGENCY_ANNOTATION
executor_future_t<E, result_of_t<ResultFactory()>>
  bulk_then_with_executor_impl(E& exec,
                               Function f,
                               ResultFactory result_factory,
                               executor_shape_t<E> shape,
                               Future& predecessor,
                               Tuple&& shared_factory_tuple,
                               detail::index_sequence<TupleIndices...>)
{
  return detail::bulk_then_execute_with_collected_result(exec, f, shape, predecessor, result_factory, agency::get<TupleIndices>(std::forward<Tuple>(shared_factory_tuple))...);
}

// this overload handles the special case where the user function returns a scope_result
// the reason we need this case cannot be handled by the overload above is because, unlike the above case,
// there is an intermediate future which must be converted to the right type of result fututre 
template<class E, class Function, size_t scope, class T, class Future, class Tuple, size_t... TupleIndices>
__AGENCY_ANNOTATION
executor_future_t<E, typename detail::scope_result_container<scope,T,E>::result_type>
  bulk_then_with_executor_impl(E& exec,
                               Function f,
                               construct<detail::scope_result_container<scope,T,E>, executor_shape_t<E>> result_factory,
                               executor_shape_t<E> shape,
                               Future& predecessor,
                               Tuple&& shared_factory_tuple,
                               detail::index_sequence<TupleIndices...>)
{
  auto intermediate_future = bulk_then_execute_with_collected_result(exec, f, shape, predecessor, result_factory, agency::get<TupleIndices>(std::forward<Tuple>(shared_factory_tuple))...);

  using result_type = typename detail::scope_result_container<scope,T,E>::result_type;

  // cast the intermediate_future to result_type
  return agency::future_cast<result_type>(exec, intermediate_future);
}

// this overload handles the special case where the user function returns void
template<class E, class Function, class Future, class Tuple, size_t... TupleIndices>
__AGENCY_ANNOTATION
executor_future_t<E, void>
  bulk_then_with_executor_impl(E& exec,
                               Function f,
                               void_factory,
                               executor_shape_t<E> shape,
                               Future& predecessor,
                               Tuple&& shared_factory_tuple,
                               detail::index_sequence<TupleIndices...>)
{
  return detail::bulk_then_execute_with_void_result(exec, f, shape, predecessor, agency::get<TupleIndices>(std::forward<Tuple>(shared_factory_tuple))...);
}


// XXX upon c++14, just return auto from these functions
template<class Result, class Future, class Function>
struct unpack_shared_parameters_from_then_execute_and_invoke
{
  mutable Function f;

  // this overload of impl() handles the case when the future given to then_execute() is non-void
  template<size_t... TupleIndices, class Index, class PastArg, class Tuple>
  __AGENCY_ANNOTATION
  Result impl(detail::index_sequence<TupleIndices...>, const Index& idx, PastArg& past_arg, Tuple&& tuple_of_shared_args) const
  {
    return f(idx, past_arg, agency::get<TupleIndices>(tuple_of_shared_args)...);
  }

  // this overload of impl() handles the case when the future given to then_execute() is void
  template<size_t... TupleIndices, class Index, class Tuple>
  __AGENCY_ANNOTATION
  Result impl(detail::index_sequence<TupleIndices...>, const Index& idx, Tuple&& tuple_of_shared_args) const
  {
    return f(idx, agency::get<TupleIndices>(tuple_of_shared_args)...);
  }

  // this overload of operator() handles the case when the future given to then_execute() is non-void
  template<class Index, class PastArg, class... Types,
           class Future1 = Future,
           class = typename std::enable_if<
             is_non_void_future<Future1>::value
           >::type>
  __AGENCY_ANNOTATION
  Result operator()(const Index& idx, PastArg& past_arg, Types&... packaged_shared_args) const
  {
    // unpack the packaged shared parameters into a tuple
    auto tuple_of_shared_args = detail::unpack_shared_parameters_from_executor(packaged_shared_args...);

    return impl(detail::make_tuple_indices(tuple_of_shared_args), idx, past_arg, tuple_of_shared_args);
  }


  // this overload of operator() handles the case when the future given to then_execute() is void
  template<class Index, class... Types,
           class Future1 = Future,
           class = typename std::enable_if<
             is_void_future<Future1>::value
           >::type>
  __AGENCY_ANNOTATION
  Result operator()(const Index& idx, Types&... packaged_shared_args) const
  {
    // unpack the packaged shared parameters into a tuple
    auto tuple_of_shared_args = detail::unpack_shared_parameters_from_executor(packaged_shared_args...);

    return impl(detail::make_tuple_indices(tuple_of_shared_args), idx, tuple_of_shared_args);
  }
};

template<class Result, class Future, class Function>
__AGENCY_ANNOTATION
unpack_shared_parameters_from_then_execute_and_invoke<Result, Future, Function> make_unpack_shared_parameters_from_then_execute_and_invoke(Function f)
{
  return unpack_shared_parameters_from_then_execute_and_invoke<Result, Future, Function>{f};
}


// computes the result type of bulk_then(executor)
template<class Executor, class Function, class Future, class... Args>
struct bulk_then_with_executor_result
{
  // figure out the Future's result_type
  using future_value_type = future_result_t<Future>;

  // assemble a list of template parameters for bulk_async_executor_result
  // when Future is a void future, we don't want to include it in the list
  using template_parameters = typename std::conditional<
    is_void_future<Future>::value,
    type_list<Executor,Function,Args...>,
    type_list<Executor,Function,Future,Args...>
  >::type;

  // to compute the result of bulk_then_executor(), instantiate
  // bulk_async_executor_result_t with the list of template parameters
  using type = type_list_instantiate<bulk_async_with_executor_result_t, template_parameters>;
};

template<class Executor, class Function, class Future, class... Args>
using bulk_then_with_executor_result_t = typename bulk_then_with_executor_result<Executor,Function,Future,Args...>::type;


template<class Future,
         class Function,
         class... Args,
         class = typename std::enable_if<
           is_non_void_future<Future>::value
         >::type>
__AGENCY_ANNOTATION
auto bind_agent_local_parameters_for_bulk_then(Function f, Args&&... args) ->
  decltype(detail::bind_agent_local_parameters_workaround_nvbug1754712(std::integral_constant<size_t,2>(), f, detail::placeholders::_1, detail::placeholders::_2, std::forward<Args>(args)...))
{
  // the _1 is for the executor idx parameter, which is the first parameter passed to f
  // the _2 is for the future parameter, which is the second parameter passed to f
  // the agent local parameters begin at index 2
  return detail::bind_agent_local_parameters_workaround_nvbug1754712(std::integral_constant<size_t,2>(), f, detail::placeholders::_1, detail::placeholders::_2, std::forward<Args>(args)...);
}

template<class Future,
         class Function,
         class... Args,
         class = typename std::enable_if<
           is_void_future<Future>::value
         >::type>
__AGENCY_ANNOTATION
auto bind_agent_local_parameters_for_bulk_then(Function f, Args&&... args) ->
  decltype(detail::bind_agent_local_parameters_workaround_nvbug1754712(std::integral_constant<size_t,1>(), f, detail::placeholders::_1, std::forward<Args>(args)...))
{
  // the _1 is for the executor idx parameter, which is the first parameter passed to f
  // the Future is void, so we don't have to reserve a parameter slot for its (non-existent) value
  // the agent local parameters begin at index 1
  return detail::bind_agent_local_parameters_workaround_nvbug1754712(std::integral_constant<size_t,1>(), f, detail::placeholders::_1, std::forward<Args>(args)...);
}


template<class Executor, class Function, class Future, class... Args>
__AGENCY_ANNOTATION
bulk_then_with_executor_result_t<Executor,Function,Future,Args...>
  bulk_then_with_executor(Executor& exec, executor_shape_t<Executor> shape, Function f, Future& fut, Args&&... args)
{
  // bind f and the agent local parameters in args... into a functor g
  auto g = detail::bind_agent_local_parameters_for_bulk_then<Future>(f, std::forward<Args>(args)...);

  // make a tuple of the shared args
  auto shared_arg_tuple = detail::forward_shared_parameters_as_tuple(std::forward<Args>(args)...);

  // package up the shared parameters for the executor
  const size_t execution_depth = executor_execution_depth<Executor>::value;

  // create a tuple of factories to use for shared parameters for the executor
  auto factory_tuple = agency::detail::make_shared_parameter_factory_tuple<execution_depth>(shared_arg_tuple);

  // compute the type of f's result
  using result_of_f = detail::result_of_continuation_t<Function,executor_index_t<Executor>,Future,decay_parameter_t<Args>...>;

  // unpack shared parameters we receive from the executor
  auto h = detail::make_unpack_shared_parameters_from_then_execute_and_invoke<result_of_f,Future>(g);

  // based on the type of f's result, make a factory that will create the appropriate type of container to store f's results
  auto result_factory = detail::make_result_factory<result_of_f>(exec, shape);

  return detail::bulk_then_with_executor_impl(exec, h, result_factory, shape, fut, factory_tuple, detail::make_index_sequence<execution_depth>());
}


} // end detail
} // end agency

