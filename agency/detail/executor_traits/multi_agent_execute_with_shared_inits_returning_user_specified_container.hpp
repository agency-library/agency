#pragma once

#include <agency/detail/config.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{
namespace multi_agent_execute_with_shared_inits_returning_user_specified_container_implementation_strategies
{


// 1.
struct use_multi_agent_execute_with_shared_inits_returning_user_specified_container_member_function {};

using use_strategy_1 = use_multi_agent_execute_with_shared_inits_returning_user_specified_container_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_1 = has_multi_agent_execute_with_shared_inits_returning_user_specified_container<Container,Executor,Function,Types...>;


// 2.
struct use_multi_agent_execute_returning_user_specified_container_member_function {};

using use_strategy_2 = use_multi_agent_execute_returning_user_specified_container_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_2 = has_multi_agent_execute_returning_user_specified_container<Container,Executor,test_function_returning_int>;


// 3.
struct use_multi_agent_execute_with_shared_inits_returning_void_member_function {};

using use_strategy_3 = use_multi_agent_execute_with_shared_inits_returning_void_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_3 = has_multi_agent_execute_with_shared_inits_returning_void<Executor,Function,Types...>;


// 4.
struct use_multi_agent_execute_returning_void_member_function {};

using use_strategy_4 = use_multi_agent_execute_returning_void_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_4 = has_multi_agent_execute_returning_void<Executor>;


// 5.
struct use_multi_agent_async_execute_with_shared_inits_returning_user_specified_container_member_function {};

using use_strategy_5 = use_multi_agent_async_execute_with_shared_inits_returning_user_specified_container_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_5 = has_multi_agent_async_execute_with_shared_inits_returning_user_specified_container<Container,Executor,Function,Types...>;


// 6.
struct use_multi_agent_then_execute_with_shared_inits_returning_user_specified_container_member_function {};

using use_strategy_6 = use_multi_agent_then_execute_with_shared_inits_returning_user_specified_container_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_6 = has_multi_agent_then_execute_with_shared_inits_returning_user_specified_container<Container,Executor,Function,typename new_executor_traits<Executor>::template future<void>, Types...>;


// 7.
struct use_multi_agent_when_all_execute_and_select_with_shared_inits_member_function {};

using use_strategy_7 = use_multi_agent_when_all_execute_and_select_with_shared_inits_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_7 = has_multi_agent_when_all_execute_and_select_with_shared_inits<
  detail::index_sequence<0>,
  Executor,
  test_function_returning_void,
  detail::tuple<
    typename new_executor_traits<Executor>::template future<Container>
  >,
  detail::type_list<Types...>
>;


// 8.
struct use_multi_agent_async_execute_returning_user_specified_container_member_function {};

using use_strategy_8 = use_multi_agent_async_execute_returning_user_specified_container_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_8 = has_multi_agent_async_execute_returning_user_specified_container<Container, Executor, Function>;


// 9.
struct use_multi_agent_then_execute_returning_user_specified_container_member_function {};

using use_strategy_9 = use_multi_agent_then_execute_returning_user_specified_container_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_9 = has_multi_agent_then_execute_returning_user_specified_container<Container, Executor, Function, typename new_executor_traits<Executor>::template future<void>>;


// 10.
struct use_multi_agent_when_all_execute_and_select_member_function {};

using use_strategy_10 = use_multi_agent_when_all_execute_and_select_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_10 = has_multi_agent_when_all_execute_and_select<
  Executor,
  test_function_returning_void,
  detail::tuple<
    typename new_executor_traits<Executor>::template future<Container>
  >,
  0
>;


// 11.
struct use_multi_agent_async_execute_with_shared_inits_returning_void_member_function {};

using use_strategy_11 = use_multi_agent_async_execute_with_shared_inits_returning_void_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_11 = has_multi_agent_async_execute_with_shared_inits_returning_void<Executor,Function,Types...>;


// 12.
struct use_multi_agent_then_execute_with_shared_inits_returning_void_member_function {};

using use_strategy_12 = use_multi_agent_then_execute_with_shared_inits_returning_void_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_12 = has_multi_agent_then_execute_with_shared_inits_returning_void<Executor,Function,typename new_executor_traits<Executor>::template future<void>,Types...>;


// 13.
struct use_multi_agent_execute_with_shared_inits_returning_default_container_member_function {};

using use_strategy_13 = use_multi_agent_execute_with_shared_inits_returning_default_container_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_13 = has_multi_agent_execute_with_shared_inits_returning_default_container<Executor, Function, Types...>;


// 14.
struct use_multi_agent_async_execute_with_shared_inits_returning_default_container_member_function {};

using use_strategy_14 = use_multi_agent_async_execute_with_shared_inits_returning_default_container_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_14 = has_multi_agent_async_execute_with_shared_inits_returning_default_container<Executor, Function, Types...>;


// 15.
struct use_multi_agent_then_execute_with_shared_inits_returning_default_container_member_function {};

using use_strategy_15 = use_multi_agent_then_execute_with_shared_inits_returning_default_container_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_15 = has_multi_agent_then_execute_with_shared_inits_returning_default_container<Executor, Function, typename new_executor_traits<Executor>::template future<void>, Types...>;


// 16.
struct use_multi_agent_async_execute_returning_void_member_function {};

using use_strategy_16 = use_multi_agent_async_execute_returning_void_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_16 = has_multi_agent_async_execute_returning_void<Executor>;


// 17.
struct use_multi_agent_then_execute_returning_void_member_function {};

using use_strategy_17 = use_multi_agent_then_execute_returning_void_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_17 = has_multi_agent_then_execute_returning_void<Executor>;


// 18.
struct use_multi_agent_execute_returning_default_container_member_function {};

using use_strategy_18 = use_multi_agent_execute_returning_default_container_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_18 = has_multi_agent_execute_returning_default_container<Executor>;


// 19.
struct use_multi_agent_async_execute_returning_default_container_member_function {};

using use_strategy_19 = use_multi_agent_async_execute_returning_default_container_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_19 = has_multi_agent_async_execute_returning_default_container<Executor>;


// 20.
struct use_multi_agent_then_execute_returning_default_container_member_function {};

using use_strategy_20 = use_multi_agent_then_execute_returning_default_container_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_20 = has_multi_agent_then_execute_returning_default_container<Executor>;


// 21.
struct use_single_agent_execute_member_function {};

using use_strategy_21 = use_single_agent_execute_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_21 = has_single_agent_execute<Executor>;


// 22.
struct use_single_agent_async_execute_member_function {};

using use_strategy_22 = use_single_agent_async_execute_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_22 = has_single_agent_async_execute<Executor>;


// 23.
struct use_single_agent_then_execute_member_function {};

using use_strategy_23 = use_single_agent_then_execute_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_23 = has_single_agent_then_execute<Executor>;


// 24.
struct use_single_agent_when_all_execute_and_select_member_function {};

using use_strategy_24 = use_single_agent_when_all_execute_and_select_member_function;

template<class Executor, class Container, class Function, class... Types>
using has_strategy_24 = has_single_agent_when_all_execute_and_select<
  Executor,
  test_function_returning_void,
  detail::tuple<
    typename new_executor_traits<Executor>::template future<int>
  >,
  0
>;


// 25.
struct use_bare_for_loop {};

using use_strategy_25 = use_bare_for_loop;


// XXX find a cleaner way to express this
template<class Executor, class Container, class Function, class... Types>
using select_multi_agent_execute_with_shared_inits_returning_user_specified_container_implementation =
  typename std::conditional<
    has_strategy_1<Executor,Container,Function,Types...>::value,
    use_strategy_1,
    typename std::conditional<
      has_strategy_2<Executor,Container,Function,Types...>::value,
      use_strategy_2,
      typename std::conditional<
        has_strategy_3<Executor,Container,Function,Types...>::value,
        use_strategy_3,
        typename std::conditional<
          has_strategy_4<Executor,Container,Function,Types...>::value,
          use_strategy_4,
          typename std::conditional<
            has_strategy_5<Executor,Container,Function,Types...>::value,
            use_strategy_5,
            typename std::conditional<
              has_strategy_6<Executor,Container,Function,Types...>::value,
              use_strategy_6,
              typename std::conditional<
                has_strategy_7<Executor,Container,Function,Types...>::value,
                use_strategy_7,
                typename std::conditional<
                  has_strategy_8<Executor,Container,Function,Types...>::value,
                  use_strategy_8,
                  typename std::conditional<
                    has_strategy_9<Executor,Container,Function,Types...>::value,
                    use_strategy_9,
                    typename std::conditional<
                      has_strategy_10<Executor,Container,Function,Types...>::value,
                      use_strategy_10,
                      typename std::conditional<
                        has_strategy_11<Executor,Container,Function,Types...>::value,
                        use_strategy_11,
                        typename std::conditional<
                          has_strategy_12<Executor,Container,Function,Types...>::value,
                          use_strategy_12,
                          typename std::conditional<
                            has_strategy_13<Executor,Container,Function,Types...>::value,
                            use_strategy_13,
                            typename std::conditional<
                              has_strategy_14<Executor,Container,Function,Types...>::value,
                              use_strategy_14,
                              typename std::conditional<
                                has_strategy_15<Executor,Container,Function,Types...>::value,
                                use_strategy_15,
                                typename std::conditional<
                                  has_strategy_16<Executor,Container,Function,Types...>::value,
                                  use_strategy_16,
                                  typename std::conditional<
                                    has_strategy_17<Executor,Container,Function,Types...>::value,
                                    use_strategy_17,
                                    typename std::conditional<
                                      has_strategy_18<Executor,Container,Function,Types...>::value,
                                      use_strategy_18,
                                      typename std::conditional<
                                        has_strategy_19<Executor,Container,Function,Types...>::value,
                                        use_strategy_19,
                                        typename std::conditional<
                                          has_strategy_20<Executor,Container,Function,Types...>::value,
                                          use_strategy_20,
                                          typename std::conditional<
                                            has_strategy_21<Executor,Container,Function,Types...>::value,
                                            use_strategy_21,
                                            typename std::conditional<
                                              has_strategy_22<Executor,Container,Function,Types...>::value,
                                              use_strategy_22,
                                              typename std::conditional<
                                                has_strategy_23<Executor,Container,Function,Types...>::value,
                                                use_strategy_23,
                                                typename std::conditional<
                                                  has_strategy_24<Executor,Container,Function,Types...>::value,
                                                  use_strategy_24,
                                                  use_strategy_25
                                                >::type
                                              >::type
                                            >::type
                                          >::type
                                        >::type
                                      >::type
                                    >::type
                                  >::type
                                >::type
                              >::type
                            >::type
                          >::type
                        >::type
                      >::type
                    >::type
                  >::type
                >::type
              >::type
            >::type
          >::type
        >::type
      >::type
    >::type
  >::type;


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_execute_with_shared_inits_returning_user_specified_container_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  return ex.template execute<Container>(f, shape, std::forward<Types>(shared_inits)...);
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Result, class Function, class Shape, class TupleOfContainers>
struct multi_agent_execute_with_shared_inits_functor
{
  mutable Function f;
  Shape shape;
  TupleOfContainers& shared_arg_containers;

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

  template<size_t... ContainerIndices, class AgentIndex>
  __AGENCY_ANNOTATION
  Result impl(detail::index_sequence<ContainerIndices...>, AgentIndex&& agent_idx) const
  {
    return f(std::forward<AgentIndex>(agent_idx),                                                      // pass the agent index
      std::get<ContainerIndices>(shared_arg_containers)[rank_in_group<ContainerIndices>(agent_idx)]... // pass the arguments coming in from shared parameters
    );
  }

  template<class Index>
  __AGENCY_ANNOTATION
  Result operator()(Index&& idx) const
  {
    static const size_t num_containers = std::tuple_size<TupleOfContainers>::value;
    return impl(detail::make_index_sequence<num_containers>(), std::forward<Index>(idx));
  }
};


template<class Result, class Function, class Shape, class TupleOfContainers>
__AGENCY_ANNOTATION
multi_agent_execute_with_shared_inits_functor<Result,Function,Shape,TupleOfContainers>
  make_multi_agent_execute_with_shared_inits_functor(Function f, Shape shape, TupleOfContainers& tuple_of_containers)
{
  return multi_agent_execute_with_shared_inits_functor<Result,Function,Shape,TupleOfContainers>{f,shape,tuple_of_containers};
} 


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_execute_returning_user_specified_container_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  // create a tuple of containers holding a shared parameter for each group
  auto shared_param_containers_tuple = make_tuple_of_shared_parameter_containers(ex, shape, std::forward<Types>(shared_inits)...);

  using index_type = typename new_executor_traits<Executor>::index_type;
  using result_type = typename std::result_of<
    Function(
      index_type,
      typename std::decay<Types>::type&...
    )
  >::type;

  // wrap f with a functor to map container elements to shared parameters
  auto g = make_multi_agent_execute_with_shared_inits_functor<result_type>(f, shape, shared_param_containers_tuple);

  return ex.template execute<Container>(g, shape);
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


struct empty_type {};


template<class Result, class Container, class Function>
struct invoke_and_store_result_to_container
{
  Container& c;
  mutable Function f;

  template<class Index, class... Args>
  __AGENCY_ANNOTATION
  Result operator()(const Index& idx, Args&... shared_args) const
  {
    // XXX should use std::invoke()
    c[idx] = f(idx, shared_args...);

    // return something easily discardable
    return Result();
  }
};


template<class Result, class Container, class Function>
__AGENCY_ANNOTATION
invoke_and_store_result_to_container<Result,Container,Function> make_invoke_and_store_result_to_container(Container& c, Function f)
{
  return invoke_and_store_result_to_container<Result,Container,Function>{c,f};
} // end make_invoke_and_store_result_to_container()


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_execute_with_shared_inits_returning_void_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  Container results(shape);

  ex.execute(make_invoke_and_store_result_to_container<void>(results,f), shape, std::forward<Types>(shared_inits)...);

  return results;
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_execute_returning_void_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  // create a tuple of containers holding a shared parameter for each group
  auto shared_param_containers_tuple = make_tuple_of_shared_parameter_containers(ex, shape, std::forward<Types>(shared_inits)...);

  using index_type = typename new_executor_traits<Executor>::index_type;
  using result_type = typename std::result_of<
    Function(
      index_type,
      typename std::decay<Types>::type&...
    )
  >::type;

  // wrap f with a functor to map container elements to shared parameters
  auto g = make_multi_agent_execute_with_shared_inits_functor<result_type>(f, shape, shared_param_containers_tuple);

  // wrap g with a functor to store the result to a container
  Container results(shape);
  auto h = make_invoke_and_store_result_to_container<void>(results, g);

  ex.execute(h, shape);

  return results;
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_async_execute_with_shared_inits_returning_user_specified_container_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  // XXX should go through executor_traits for the get()
  return ex.template async_execute<Container>(f, shape, std::forward<Types>(shared_inits)...).get();
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_then_execute_with_shared_inits_returning_user_specified_container_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  auto ready = new_executor_traits<Executor>::template make_ready_future<void>(ex);

  // XXX should go through executor_traits for the get()
  return ex.template then_execute<Container>(f, shape, ready, std::forward<Types>(shared_inits)...).get();
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Function>
struct strategy_7_functor
{
  mutable Function f;

  template<class Index, class Container, class... Args>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Container& results, Args&... shared_args) const
  {
    // XXX should use std::invoke
    results[idx] = f(idx, shared_args...);
  }
};


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_when_all_execute_and_select_with_shared_inits_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  auto results = new_executor_traits<Executor>::template make_ready_future<Container>(ex, shape);

  auto futures = detail::make_tuple(std::move(results));

  // XXX should go through executor_traits for the get()
  return ex.template when_all_execute_and_select<0>(strategy_7_functor<Function>{f}, shape, futures, std::forward<Types>(shared_inits)...).get();
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_async_execute_returning_user_specified_container_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  // create a tuple of containers holding a shared parameter for each group
  auto shared_param_containers_tuple = make_tuple_of_shared_parameter_containers(ex, shape, std::forward<Types>(shared_inits)...);

  using index_type = typename new_executor_traits<Executor>::index_type;
  using result_type = typename std::result_of<
    Function(
      index_type,
      typename std::decay<Types>::type&...
    )
  >::type;

  // wrap f with a functor to map container elements to shared parameters
  auto g = make_multi_agent_execute_with_shared_inits_functor<result_type>(f, shape, shared_param_containers_tuple);

  // XXX should go through executor_traits for this get()
  return ex.template async_execute<Container>(g, shape).get();
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_then_execute_returning_user_specified_container_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  // create a tuple of containers holding a shared parameter for each group
  auto shared_param_containers_tuple = make_tuple_of_shared_parameter_containers(ex, shape, std::forward<Types>(shared_inits)...);

  using index_type = typename new_executor_traits<Executor>::index_type;
  using result_type = typename std::result_of<
    Function(
      index_type,
      typename std::decay<Types>::type&...
    )
  >::type;

  // wrap f with a functor to map container elements to shared parameters
  auto g = make_multi_agent_execute_with_shared_inits_functor<result_type>(f, shape, shared_param_containers_tuple);

  auto ready = new_executor_traits<Executor>::template make_ready_future<void>(ex);

  // XXX should go through executor_traits for this get()
  return ex.template then_execute<Container>(g, shape, ready).get();
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Function>
struct invoke_and_store_to_second_parameter
{
  mutable Function f;

  template<class Index, class Container>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Container& c) const
  {
    // XXX should use std::invoke()
    c[idx] = f(idx);
  }
};

template<class Function>
__AGENCY_ANNOTATION
invoke_and_store_to_second_parameter<Function> make_invoke_and_store_to_second_parameter(Function f)
{
  return invoke_and_store_to_second_parameter<Function>{f};
}


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_when_all_execute_and_select_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  // create a tuple of containers holding a shared parameter for each group
  auto shared_param_containers_tuple = make_tuple_of_shared_parameter_containers(ex, shape, std::forward<Types>(shared_inits)...);

  using index_type = typename new_executor_traits<Executor>::index_type;
  using result_type = typename std::result_of<
    Function(
      index_type,
      typename std::decay<Types>::type&...
    )
  >::type;

  // wrap f with a functor to map container elements to shared parameters
  auto g = make_multi_agent_execute_with_shared_inits_functor<result_type>(f, shape, shared_param_containers_tuple);

  // wrap g with a functor to store g's result to the container passed as the second parameter
  auto h = make_invoke_and_store_to_second_parameter(g);

  auto results = new_executor_traits<Executor>::template make_ready_future<Container>(ex, shape);

  auto futures = detail::make_tuple(std::move(results));

  // XXX should go through executor_traits for the get()
  return ex.template when_all_execute_and_select<0>(h, shape, futures).get();
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_async_execute_with_shared_inits_returning_void_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  Container results(shape);

  // XXX should call wait() through executor_traits
  ex.async_execute(make_invoke_and_store_result_to_container<void>(results, f), shape, std::forward<Types>(shared_inits)...).wait();

  return results;
}


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_then_execute_with_shared_inits_returning_void_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  Container results(shape);

  auto ready = new_executor_traits<Executor>::template make_ready_future<void>(ex);

  // XXX should call wait() through executor_traits
  ex.then_execute(make_invoke_and_store_result_to_container<void>(results, f), shape, ready, std::forward<Types>(shared_inits)...).wait();

  return results;
}


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_execute_with_shared_inits_returning_default_container_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)

{
  Container results(shape);

  // discard the container of results returned by this call
  ex.execute(make_invoke_and_store_result_to_container<empty_type>(results,f), shape, std::forward<Types>(shared_inits)...);

  return results;
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_async_execute_with_shared_inits_returning_default_container_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  Container results(shape);

  // discard the container of results returned by this call
  // XXX should call wait() through executor_traits
  ex.async_execute(make_invoke_and_store_result_to_container<empty_type>(results,f), shape, std::forward<Types>(shared_inits)...).wait();

  return results;
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_async_execute_returning_void_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  // create a tuple of containers holding a shared parameter for each group
  auto shared_param_containers_tuple = make_tuple_of_shared_parameter_containers(ex, shape, std::forward<Types>(shared_inits)...);

  using index_type = typename new_executor_traits<Executor>::index_type;
  using result_type = typename std::result_of<
    Function(
      index_type,
      typename std::decay<Types>::type&...
    )
  >::type;

  // wrap f with a functor to map container elements to shared parameters
  auto g = make_multi_agent_execute_with_shared_inits_functor<result_type>(f, shape, shared_param_containers_tuple);

  Container results(shape);

  // wrap g with a functor to store g's results to a container
  auto h = make_invoke_and_store_result_to_container<void>(results, g);

  // discard the container returned by this call
  // XXX wait() should be called through executor_traits
  ex.async_execute(h, shape).wait();

  return results;
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_then_execute_returning_void_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  // create a tuple of containers holding a shared parameter for each group
  auto shared_param_containers_tuple = make_tuple_of_shared_parameter_containers(ex, shape, std::forward<Types>(shared_inits)...);

  using index_type = typename new_executor_traits<Executor>::index_type;
  using result_type = typename std::result_of<
    Function(
      index_type,
      typename std::decay<Types>::type&...
    )
  >::type;

  // wrap f with a functor to map container elements to shared parameters
  auto g = make_multi_agent_execute_with_shared_inits_functor<result_type>(f, shape, shared_param_containers_tuple);

  Container results(shape);

  // wrap g with a functor to store g's results to a container
  auto h = make_invoke_and_store_result_to_container<void>(results, g);

  auto ready = new_executor_traits<Executor>::template make_ready_future<void>(ex);

  // discard the container returned by this call
  // XXX wait() should be called through executor_traits
  ex.then_execute(h, shape, ready).wait();

  return results;
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_then_execute_with_shared_inits_returning_default_container_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  Container results(shape);

  auto ready = new_executor_traits<Executor>::template make_ready_future<void>(ex);

  // discard the container of results returned by this call
  // XXX should call wait() through executor_traits
  ex.then_execute(make_invoke_and_store_result_to_container<empty_type>(results,f), shape, ready, std::forward<Types>(shared_inits)...).wait();

  return results;
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_execute_returning_default_container_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  // create a tuple of containers holding a shared parameter for each group
  auto shared_param_containers_tuple = make_tuple_of_shared_parameter_containers(ex, shape, std::forward<Types>(shared_inits)...);

  using index_type = typename new_executor_traits<Executor>::index_type;
  using result_type = typename std::result_of<
    Function(
      index_type,
      typename std::decay<Types>::type&...
    )
  >::type;

  // wrap f with a functor to map container elements to shared parameters
  auto g = make_multi_agent_execute_with_shared_inits_functor<result_type>(f, shape, shared_param_containers_tuple);

  Container results(shape);

  // wrap g with a functor to store g's results to a container
  auto h = make_invoke_and_store_result_to_container<empty_type>(results, g);

  // discard the container returned by this call
  ex.execute(h, shape);

  return results;
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_async_execute_returning_default_container_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  // create a tuple of containers holding a shared parameter for each group
  auto shared_param_containers_tuple = make_tuple_of_shared_parameter_containers(ex, shape, std::forward<Types>(shared_inits)...);

  using index_type = typename new_executor_traits<Executor>::index_type;
  using result_type = typename std::result_of<
    Function(
      index_type,
      typename std::decay<Types>::type&...
    )
  >::type;

  // wrap f with a functor to map container elements to shared parameters
  auto g = make_multi_agent_execute_with_shared_inits_functor<result_type>(f, shape, shared_param_containers_tuple);

  Container results(shape);

  // wrap g with a functor to store g's results to a container
  auto h = make_invoke_and_store_result_to_container<empty_type>(results, g);

  // discard the container returned by this call
  // XXX should call wait() through executor_traits
  ex.async_execute(h, shape).wait();

  return results;
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_then_execute_returning_default_container_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  // create a tuple of containers holding a shared parameter for each group
  auto shared_param_containers_tuple = make_tuple_of_shared_parameter_containers(ex, shape, std::forward<Types>(shared_inits)...);

  using index_type = typename new_executor_traits<Executor>::index_type;
  using result_type = typename std::result_of<
    Function(
      index_type,
      typename std::decay<Types>::type&...
    )
  >::type;

  // wrap f with a functor to map container elements to shared parameters
  auto g = make_multi_agent_execute_with_shared_inits_functor<result_type>(f, shape, shared_param_containers_tuple);

  Container results(shape);

  // wrap g with a functor to store g's results to a container
  auto h = make_invoke_and_store_result_to_container<empty_type>(results, g);

  auto ready = new_executor_traits<Executor>::template make_ready_future<void>(ex);

  // discard the container returned by this call
  // XXX should call wait() through executor_traits
  ex.then_execute(h, shape, ready).wait();

  return results;
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Container, class Function, class Shape, class T>
struct execute_in_for_loop
{
  mutable Function f;
  Shape shape;
  mutable T shared_arg;

  __AGENCY_ANNOTATION
  Container operator()() const
  {
    Container results(shape);

    // XXX generalize to multidimensions
    for(size_t idx = 0; idx < shape; ++idx)
    {
      // XXX use std::invoke
      results[idx] = f(idx, shared_arg);
    }

    return results;
  }
};

template<class Container, class Function, class Shape, class T>
__AGENCY_ANNOTATION
execute_in_for_loop<Container,Function,Shape,typename std::decay<T>::type> make_execute_in_for_loop(Function f, Shape shape, T&& shared_init)
{
  return execute_in_for_loop<Container,Function,Shape,typename std::decay<T>::type>{f, shape, std::forward<T>(shared_init)};
}


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_single_agent_execute_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  static_assert(sizeof...(Types) == 1, "This implementation only makes sense for flat (i.e., single-agent) executors");

  return ex.execute(make_execute_in_for_loop<Container>(f, shape, std::forward<Types>(shared_inits)...));
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_single_agent_async_execute_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  static_assert(sizeof...(Types) == 1, "This implementation only makes sense for flat (i.e., single-agent) executors");

  // XXX should call get() through executor_traits
  return ex.async_execute(make_execute_in_for_loop<Container>(f, shape, std::forward<Types>(shared_inits)...)).get();
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_single_agent_then_execute_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  static_assert(sizeof...(Types) == 1, "This implementation only makes sense for flat (i.e., single-agent) executors");

  auto ready = new_executor_traits<Executor>::template make_ready_future<void>(ex);

  // XXX should call get() through executor_traits
  return ex.then_execute(make_execute_in_for_loop<Container>(f, shape, std::forward<Types>(shared_inits)...), ready).get();
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Container, class Function, class Shape, class T>
struct single_agent_when_all_execute_and_select_functor
{
  mutable execute_in_for_loop<Container,Function,Shape,T> implementation;

  __AGENCY_ANNOTATION
  void operator()(Container& results) const
  {
    results = implementation();
  }
};


template<class Container, class Function, class Shape, class T>
__AGENCY_ANNOTATION
single_agent_when_all_execute_and_select_functor<Container,Function,Shape,typename std::decay<T>::type> make_single_agent_when_all_execute_and_select_functor(Function f, Shape shape, T&& shared_init)
{
  return single_agent_when_all_execute_and_select_functor<Container,Function,Shape,typename std::decay<T>::type>{
    make_execute_in_for_loop<Container>(f, shape, std::forward<T>(shared_init))
  };
}


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_single_agent_when_all_execute_and_select_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  static_assert(sizeof...(Types) == 1, "This implementation only makes sense for flat (i.e., single-agent) executors");

  auto results = new_executor_traits<Executor>::template make_ready_future<Container>(ex);
  auto tuple_of_futures = detail::make_tuple(std::move(results));

  // XXX should call get() through executor_traits
  return ex.template when_all_execute_and_select<0>(make_single_agent_when_all_execute_and_select_functor<Container>(f, shape, std::forward<Types>(shared_inits)...), tuple_of_futures).get();
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_bare_for_loop,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  static_assert(sizeof...(Types) == 1, "This implementation only makes sense for flat (i.e., single-agent) executors");

  auto implementation = make_execute_in_for_loop<Container>(f, shape, std::forward<Types>(shared_inits)...);
  return implementation();
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


} // end multi_agent_execute_with_shared_inits_returning_user_specified_container_implementation_strategies
} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Container, class Function, class... Types,
           class Enable>
Container new_executor_traits<Executor>
  ::execute(typename new_executor_traits<Executor>::executor_type& ex,
            Function f,
            typename new_executor_traits<Executor>::shape_type shape,
            Types&&... shared_inits)
{
  namespace ns = detail::new_executor_traits_detail::multi_agent_execute_with_shared_inits_returning_user_specified_container_implementation_strategies;

  using implementation_strategy = ns::select_multi_agent_execute_with_shared_inits_returning_user_specified_container_implementation<
    Executor,
    Container,
    Function,
    Types&&...
  >;

  return ns::multi_agent_execute_with_shared_inits_returning_user_specified_container<Container>(implementation_strategy(), ex, f, shape, std::forward<Types>(shared_inits)...);
} // end new_executor_traits::execute()


} // end agency

