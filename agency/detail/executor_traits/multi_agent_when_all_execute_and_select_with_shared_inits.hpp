#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/index_cast.hpp>
#include <type_traits>
#include <utility>
#include <cassert>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<size_t... Indices, class Executor, class TupleOfFutures, class Function, class... Types>
typename new_executor_traits<Executor>::template future<
  detail::when_all_execute_and_select_result_t<
    detail::index_sequence<Indices...>,
    typename std::decay<TupleOfFutures>::type
  >
>
  multi_agent_when_all_execute_and_select_with_shared_inits(std::true_type, Executor& ex, TupleOfFutures&& futures, Function f, typename new_executor_traits<Executor>::shape_type shape, Types&&... shared_inits)
{
  return ex.template when_all_execute_and_select<Indices...>(std::forward<TupleOfFutures>(futures), f, shape, std::forward<Types>(shared_inits)...);
} // end multi_agent_when_all_execute_and_select_with_shared_inits()


template<class Function, class Shape>
struct multi_agent_when_all_execute_and_select_with_shared_inits_functor
{
  mutable Function f;
  Shape shape;

  template<size_t depth, class AgentIndex>
  __AGENCY_ANNOTATION
  size_t rank_in_group(const AgentIndex& idx) const
  {
    // to compute the rank of an index at a particular depth,
    // first prepend 0 (1) to idx (shape) to represent an index of the root group (it has none otherwise)
    // XXX seems like index_cast() should just do the right thing for empty indices
    //     it would correspond to a single-agent task
    auto augmented_idx   = detail::tuple_prepend(detail::wrap_scalar(idx), size_t{0});
    auto augmented_shape = detail::tuple_prepend(detail::wrap_scalar(shape), size_t{1});
    
    // take the first depth+1 (plus one because we prepended 1) indices of the index & shape and do an index_cast to size_t
    return detail::index_cast<size_t>(detail::tuple_take<depth+1>(augmented_idx),
                                      detail::tuple_take<depth+1>(augmented_shape),
                                      detail::shape_size(detail::tuple_take<depth+1>(augmented_shape)));
  }

  // XXX generalize this to handle a variadic number of past args
  template<size_t... ContainerIndices, class T1, class TupleOfContainers, class AgentIndex>
  __AGENCY_ANNOTATION
  void impl(detail::index_sequence<ContainerIndices...>, T1& past_arg, TupleOfContainers& shared_arg_containers, AgentIndex&& agent_idx) const
  {
    f(past_arg, std::forward<AgentIndex>(agent_idx), std::get<ContainerIndices>(shared_arg_containers)[rank_in_group<ContainerIndices>(agent_idx)]...);
  }

  // XXX generalize this to handle a variadic number of past args
  template<class T1, class TupleOfContainers, class Index>
  __AGENCY_ANNOTATION
  void operator()(T1& past_arg, TupleOfContainers& shared_arg_containers, Index&& idx) const
  {
    static const size_t num_containers = std::tuple_size<TupleOfContainers>::value;

    impl(detail::make_index_sequence<num_containers>(), past_arg, shared_arg_containers, std::forward<Index>(idx));
  }
};


template<size_t depth, class Shape>
size_t number_of_groups_at_depth(const Shape& shape)
{
  // to compute the number of groups at a particular depth given a shape,
  // take the first depth elements of shape and return shape_size
  return detail::shape_size(detail::tuple_take<depth>(shape));
}


template<class T, class Executor>
using shared_parameter_container = typename new_executor_traits<Executor>::template container<T>;


template<class Executor, class T>
shared_parameter_container<T,Executor> make_shared_parameter_container(Executor&, size_t n, const T& shared_init)
{
  return shared_parameter_container<T,Executor>(n, shared_init);
}


template<size_t... Indices, class Executor, class... Types>
detail::tuple<shared_parameter_container<Types,Executor>...>
  make_tuple_of_shared_parameter_containers(detail::index_sequence<Indices...>, Executor& ex, typename new_executor_traits<Executor>::shape_type shape, const Types&... shared_inits)
{
  return detail::make_tuple(make_shared_parameter_container(ex, number_of_groups_at_depth<Indices>(shape), shared_inits)...);
}


template<class Executor, class... Types>
detail::tuple<
  shared_parameter_container<
    typename std::decay<Types>::type,
    Executor
  >...
>
  make_tuple_of_shared_parameter_containers(Executor& ex, typename new_executor_traits<Executor>::shape_type shape, Types&&... shared_inits)
{
  return make_tuple_of_shared_parameter_containers(detail::make_index_sequence<sizeof...(shared_inits)>(), ex, shape, std::forward<Types>(shared_inits)...);
}


template<size_t... Indices, class Executor, class TupleOfFutures, class Function, class... Types>
typename new_executor_traits<Executor>::template future<
  detail::when_all_execute_and_select_result_t<
    detail::index_sequence<Indices...>,
    typename std::decay<TupleOfFutures>::type
  >
>
  multi_agent_when_all_execute_and_select_with_shared_inits(std::false_type, Executor& ex, TupleOfFutures&& futures, Function f, typename new_executor_traits<Executor>::shape_type shape, Types&&... shared_inits)
{
  // create a tuple of containers holding a shared parameter for each group
  auto shared_param_containers_tuple = make_tuple_of_shared_parameter_containers(ex, shape, std::forward<Types>(shared_inits)...);

  // turn it into a future
  auto shared_param_containers_tuple_fut = new_executor_traits<Executor>::template make_ready_future<decltype(shared_param_containers_tuple)>(ex, std::move(shared_param_containers_tuple));

  // combine the shared parameters with the incoming futures
  auto futures_and_shared = detail::tuple_append(std::move(futures), std::move(shared_param_containers_tuple_fut));

  // wrap f with a functor to map container elements to shared parameters
  auto g = multi_agent_when_all_execute_and_select_with_shared_inits_functor<Function, typename new_executor_traits<Executor>::shape_type>{f, shape};

  return new_executor_traits<Executor>::template when_all_execute_and_select<Indices...>(ex, std::move(futures_and_shared), g, shape);
} // end multi_agent_when_all_execute_and_select_with_shared_inits()


template<class IndexSequence, class Executor, class TupleOfFutures, class Function, class Shape, class TypeList>
struct has_multi_agent_when_all_execute_and_select_with_shared_inits_impl;


template<size_t... Indices, class Executor, class TupleOfFutures, class Function, class Shape, class... Types>
struct has_multi_agent_when_all_execute_and_select_with_shared_inits_impl<index_sequence<Indices...>, Executor, TupleOfFutures, Function, Shape, type_list<Types...>>
{
  template<class Executor1,
           class = decltype(
             std::declval<Executor1>().template when_all_execute_and_select<Indices...>(
               std::declval<TupleOfFutures>(),
               std::declval<Function>(),
               std::declval<Shape>(),
               std::declval<Types>()...
             )
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class IndexSequence, class Executor, class TupleOfFutures, class Function, class Shape, class TypeList>
using has_multi_agent_when_all_execute_and_select_with_shared_inits = typename has_multi_agent_when_all_execute_and_select_with_shared_inits_impl<IndexSequence, Executor, TupleOfFutures, Function, Shape, TypeList>::type;


} // end detail
} // end new_executor_traits_detail


template<class Executor>
template<size_t... Indices, class TupleOfFutures, class Function, class T1, class... Types>
  typename new_executor_traits<Executor>::template future<
    detail::when_all_execute_and_select_result_t<
      detail::index_sequence<Indices...>,
      typename std::decay<TupleOfFutures>::type
    >
  >
  new_executor_traits<Executor>
    ::when_all_execute_and_select(typename new_executor_traits<Executor>::executor_type& ex,
                                  TupleOfFutures&& futures,
                                  Function f,
                                  typename new_executor_traits<Executor>::shape_type shape,
                                  T1&& outer_shared_init,
                                  Types&&... inner_shared_inits)
{
  static_assert(new_executor_traits<Executor>::execution_depth == 1 + sizeof...(Types), "The number of shared initializers must be equal to the executor's execution_depth.");
  
  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_when_all_execute_and_select_with_shared_inits<
    detail::index_sequence<Indices...>,
    Executor,
    typename std::decay<TupleOfFutures>::type,
    Function,
    typename new_executor_traits<Executor>::shape_type,
    detail::type_list<T1,Types...>
  >;

  return detail::new_executor_traits_detail::multi_agent_when_all_execute_and_select_with_shared_inits<Indices...>(check_for_member_function(), ex, std::forward<TupleOfFutures>(futures), f, shape, std::forward<T1>(outer_shared_init), std::forward<Types>(inner_shared_inits)...);
} // end new_executor_traits::when_all_execute_and_select()


} // end agency

