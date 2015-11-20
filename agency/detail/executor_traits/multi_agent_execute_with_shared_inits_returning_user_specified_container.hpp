#pragma once

#include <agency/detail/config.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/functional.hpp>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{
namespace new_multi_agent_execute_with_shared_inits_returning_user_specified_container_implementation_strategies
{


// 1.
struct use_multi_agent_execute_with_shared_inits_returning_user_specified_container_member_function {};

using use_strategy_1 = use_multi_agent_execute_with_shared_inits_returning_user_specified_container_member_function;

template<class Executor, class Function, class Factory, class... Factories>
using has_strategy_1 = has_multi_agent_execute_with_shared_inits_returning_user_specified_container<Executor,Function,Factory,Factories...>;

template<class Executor, class Function, class Factory, class... Factories>
struct has_strategy_1_workaround_nvbug_1665745
{
  using type = has_multi_agent_execute_with_shared_inits_returning_user_specified_container<Executor,Function,Factory,Factories...>;
};


// 2.
struct use_multi_agent_execute_returning_user_specified_container_member_function {};

using use_strategy_2 = use_multi_agent_execute_returning_user_specified_container_member_function;

template<class Executor, class Function, class Factory, class... Factories>
using has_strategy_2 = has_multi_agent_execute_returning_user_specified_container<Executor,test_function_returning_int,Factory>;


// 3.
struct use_multi_agent_execute_with_shared_inits_returning_void_member_function {};

using use_strategy_3 = use_multi_agent_execute_with_shared_inits_returning_void_member_function;

template<class Executor, class Function, class Factory, class... Factories>
using has_strategy_3 = has_multi_agent_execute_with_shared_inits_returning_void<Executor,Function,Factories...>;

template<class Executor, class Function, class Factory, class... Factories>
struct has_strategy_3_workaround_nvbug_1665745
{
  using type = has_multi_agent_execute_with_shared_inits_returning_void<Executor,Function,Factories...>;
};


// 4.
struct use_multi_agent_execute_returning_void_member_function {};

using use_strategy_4 = use_multi_agent_execute_returning_void_member_function;

template<class Executor, class Function, class Factory, class... Factories>
using has_strategy_4 = has_multi_agent_execute_returning_void<Executor>;


// 5.
struct use_multi_agent_async_execute_with_shared_inits_returning_user_specified_container_member_function {};

using use_strategy_5 = use_multi_agent_async_execute_with_shared_inits_returning_user_specified_container_member_function;

template<class Executor, class Function, class Factory, class... Factories>
using has_strategy_5 = has_multi_agent_async_execute_with_shared_inits_returning_user_specified_container<Executor,Function,Factory,Factories...>;

template<class Executor, class Function, class Factory, class... Factories>
struct has_strategy_5_workaround_nvbug_1665745
{
  using type = has_multi_agent_async_execute_with_shared_inits_returning_user_specified_container<Executor,Function,Factory,Factories...>;
};


// 6.
struct use_multi_agent_then_execute_with_shared_inits_returning_user_specified_container_member_function {};

using use_strategy_6 = use_multi_agent_then_execute_with_shared_inits_returning_user_specified_container_member_function;

template<class Executor, class Function, class Factory, class... Factories>
using has_strategy_6 = has_multi_agent_then_execute_with_shared_inits_returning_user_specified_container<Executor,Function,Factory,typename new_executor_traits<Executor>::template future<void>, Factories...>;

template<class Executor, class Function, class Factory, class... Factories>
struct has_strategy_6_workaround_nvbug_1665745
{
  using type = has_multi_agent_then_execute_with_shared_inits_returning_user_specified_container<Executor,Function,Factory,typename new_executor_traits<Executor>::template future<void>, Factories...>;
};


// 7.
struct use_multi_agent_when_all_execute_and_select_with_shared_inits_member_function {};

using use_strategy_7 = use_multi_agent_when_all_execute_and_select_with_shared_inits_member_function;

template<class Executor, class Function, class Factory, class... Factories>
using has_strategy_7 = has_multi_agent_when_all_execute_and_select_with_shared_inits<
  detail::index_sequence<0>,
  Executor,
  test_function_returning_void,
  detail::tuple<
    typename new_executor_traits<Executor>::template future<
      typename std::result_of<
        Factory(typename new_executor_traits<Executor>::shape_type)
      >::type
    >
  >,
  detail::type_list<Factories...>
>;


// 8.
struct use_multi_agent_async_execute_returning_user_specified_container_member_function {};

using use_strategy_8 = use_multi_agent_async_execute_returning_user_specified_container_member_function;

template<class Executor, class Function, class Factory, class... Factories>
using has_strategy_8 = has_multi_agent_async_execute_returning_user_specified_container<Executor, Function, Factory>;


// 9.
struct use_multi_agent_then_execute_returning_user_specified_container_member_function {};

using use_strategy_9 = use_multi_agent_then_execute_returning_user_specified_container_member_function;

template<class Executor, class Function, class Factory, class... Factories>
using has_strategy_9 = has_multi_agent_then_execute_returning_user_specified_container<Executor, Function, Factory, typename new_executor_traits<Executor>::template future<void>>;


// 10.
struct use_multi_agent_when_all_execute_and_select_member_function {};

using use_strategy_10 = use_multi_agent_when_all_execute_and_select_member_function;

template<class Executor, class Function, class Factory, class... Factories>
using has_strategy_10 = has_multi_agent_when_all_execute_and_select<
  Executor,
  test_function_returning_void,
  detail::tuple<
    typename new_executor_traits<Executor>::template future<
      typename std::result_of<
        Factory(typename new_executor_traits<Executor>::shape_type)
      >::type
    >
  >,
  0
>;


// 11.
struct use_multi_agent_async_execute_with_shared_inits_returning_void_member_function {};

using use_strategy_11 = use_multi_agent_async_execute_with_shared_inits_returning_void_member_function;

template<class Executor, class Function, class Factory, class... Factories>
using has_strategy_11 = has_multi_agent_async_execute_with_shared_inits_returning_void<Executor,Function,Factories...>;

template<class Executor, class Function, class Factory, class... Factories>
struct has_strategy_11_workaround_nvbug_1665745
{
  using type = has_multi_agent_async_execute_with_shared_inits_returning_void<Executor,Function,Factories...>;
};


// 12.
struct use_multi_agent_then_execute_with_shared_inits_returning_void_member_function {};

using use_strategy_12 = use_multi_agent_then_execute_with_shared_inits_returning_void_member_function;

template<class Executor, class Function, class Factory, class... Factories>
using has_strategy_12 = has_multi_agent_then_execute_with_shared_inits_returning_void<Executor,Function,typename new_executor_traits<Executor>::template future<void>,Factories...>;

template<class Executor, class Function, class Factory, class... Factories>
struct has_strategy_12_workaround_nvbug_1665745
{
  using type = has_multi_agent_then_execute_with_shared_inits_returning_void<Executor,Function,typename new_executor_traits<Executor>::template future<void>,Factories...>;
};


// 13.
struct use_multi_agent_execute_with_shared_inits_returning_default_container_member_function {};

using use_strategy_13 = use_multi_agent_execute_with_shared_inits_returning_default_container_member_function;

template<class Executor, class Function, class Factory, class... Factories>
using has_strategy_13 = has_multi_agent_execute_with_shared_inits_returning_default_container<Executor, Function, Factories...>;

template<class Executor, class Function, class Factory, class... Factories>
struct has_strategy_13_workaround_nvbug_1665745
{
  using type = has_multi_agent_execute_with_shared_inits_returning_default_container<Executor, Function, Factories...>;
};


// 14.
struct use_multi_agent_async_execute_with_shared_inits_returning_default_container_member_function {};

using use_strategy_14 = use_multi_agent_async_execute_with_shared_inits_returning_default_container_member_function;

template<class Executor, class Function, class Factory, class... Factories>
using has_strategy_14 = has_multi_agent_async_execute_with_shared_inits_returning_default_container<Executor, Function, Factories...>;

template<class Executor, class Function, class Factory, class... Factories>
struct has_strategy_14_workaround_nvbug_1665745
{
  using type = has_multi_agent_async_execute_with_shared_inits_returning_default_container<Executor, Function, Factories...>;
};


// 15.
struct use_multi_agent_then_execute_with_shared_inits_returning_default_container_member_function {};

using use_strategy_15 = use_multi_agent_then_execute_with_shared_inits_returning_default_container_member_function;

template<class Executor, class Function, class Factory, class... Factories>
using has_strategy_15 = has_multi_agent_then_execute_with_shared_inits_returning_default_container<Executor, Function, typename new_executor_traits<Executor>::template future<void>, Factories...>;

template<class Executor, class Function, class Factory, class... Factories>
struct has_strategy_15_workaround_nvbug_1665745
{
  using type = has_multi_agent_then_execute_with_shared_inits_returning_default_container<Executor, Function, typename new_executor_traits<Executor>::template future<void>, Factories...>;
};


// 16.
struct use_multi_agent_async_execute_returning_void_member_function {};

using use_strategy_16 = use_multi_agent_async_execute_returning_void_member_function;

template<class Executor, class Function, class Factory, class... Factories>
using has_strategy_16 = has_multi_agent_async_execute_returning_void<Executor>;


// 17.
struct use_multi_agent_then_execute_returning_void_member_function {};

using use_strategy_17 = use_multi_agent_then_execute_returning_void_member_function;

template<class Executor, class Function, class Factory, class... Factories>
using has_strategy_17 = has_multi_agent_then_execute_returning_void<Executor>;


// 18.
struct use_multi_agent_execute_returning_default_container_member_function {};

using use_strategy_18 = use_multi_agent_execute_returning_default_container_member_function;

template<class Executor, class Function, class Factory, class... Factories>
using has_strategy_18 = has_multi_agent_execute_returning_default_container<Executor>;


// 19.
struct use_multi_agent_async_execute_returning_default_container_member_function {};

using use_strategy_19 = use_multi_agent_async_execute_returning_default_container_member_function;

template<class Executor, class Function, class Factory, class... Factories>
using has_strategy_19 = has_multi_agent_async_execute_returning_default_container<Executor>;


// 20.
struct use_multi_agent_then_execute_returning_default_container_member_function {};

using use_strategy_20 = use_multi_agent_then_execute_returning_default_container_member_function;

template<class Executor, class Function, class Factory, class... Factories>
using has_strategy_20 = has_multi_agent_then_execute_returning_default_container<Executor>;


// 21.
struct use_single_agent_execute_member_function {};

using use_strategy_21 = use_single_agent_execute_member_function;

template<class Executor, class Function, class Factory, class... Factories>
using has_strategy_21 = has_single_agent_execute<Executor>;


// 22.
struct use_single_agent_async_execute_member_function {};

using use_strategy_22 = use_single_agent_async_execute_member_function;

template<class Executor, class Function, class Factory, class... Factories>
using has_strategy_22 = has_single_agent_async_execute<Executor>;


// 23.
struct use_single_agent_then_execute_member_function {};

using use_strategy_23 = use_single_agent_then_execute_member_function;

template<class Executor, class Function, class Factory, class... Factories>
using has_strategy_23 = has_single_agent_then_execute<Executor>;


// 24.
struct use_single_agent_when_all_execute_and_select_member_function {};

using use_strategy_24 = use_single_agent_when_all_execute_and_select_member_function;

template<class Executor, class Function, class Factory, class... Factories>
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
template<class Executor, class Function, class Factory, class... Factories>
using select_multi_agent_execute_with_shared_inits_returning_user_specified_container_implementation =
  typename std::conditional<
    //has_strategy_1<Executor,Function,Factory,Factories...>::value,
    has_strategy_1_workaround_nvbug_1665745<Executor,Function,Factory,Factories...>::type::value,
    use_strategy_1,
    typename std::conditional<
      has_strategy_2<Executor,Function,Factory,Factories...>::value,
      use_strategy_2,
      typename std::conditional<
        //has_strategy_3<Executor,Function,Factory,Factories...>::value,
        has_strategy_3_workaround_nvbug_1665745<Executor,Function,Factory,Factories...>::type::value,
        use_strategy_3,
        typename std::conditional<
          has_strategy_4<Executor,Function,Factory,Factories...>::value,
          use_strategy_4,
          typename std::conditional<
            //has_strategy_5<Executor,Function,Factory,Factories...>::value,
            has_strategy_5_workaround_nvbug_1665745<Executor,Function,Factory,Factories...>::type::value,
            use_strategy_5,
            typename std::conditional<
              //has_strategy_6<Executor,Function,Factory,Factories...>::value,
              has_strategy_6_workaround_nvbug_1665745<Executor,Function,Factory,Factories...>::type::value,
              use_strategy_6,
              typename std::conditional<
                has_strategy_7<Executor,Function,Factory,Factories...>::value,
                use_strategy_7,
                typename std::conditional<
                  has_strategy_8<Executor,Function,Factory,Factories...>::value,
                  use_strategy_8,
                  typename std::conditional<
                    has_strategy_9<Executor,Function,Factory,Factories...>::value,
                    use_strategy_9,
                    typename std::conditional<
                      has_strategy_10<Executor,Function,Factory,Factories...>::value,
                      use_strategy_10,
                      typename std::conditional<
                        //has_strategy_11<Executor,Function,Factory,Factories...>::value,
                        has_strategy_11_workaround_nvbug_1665745<Executor,Function,Factory,Factories...>::type::value,
                        use_strategy_11,
                        typename std::conditional<
                          //has_strategy_12<Executor,Function,Factory,Factories...>::value,
                          has_strategy_12_workaround_nvbug_1665745<Executor,Function,Factory,Factories...>::type::value,
                          use_strategy_12,
                          typename std::conditional<
                            //has_strategy_13<Executor,Container,Function,Factory,Factories...>::value,
                            has_strategy_13_workaround_nvbug_1665745<Executor,Function,Factory,Factories...>::type::value,
                            use_strategy_13,
                            typename std::conditional<
                              //has_strategy_14<Executor,Function,Factory,Factories...>::value,
                              has_strategy_14_workaround_nvbug_1665745<Executor,Function,Factory,Factories...>::type::value,
                              use_strategy_14,
                              typename std::conditional<
                                //has_strategy_15<Executor,Function,Factory,Factories...>::value,
                                has_strategy_15_workaround_nvbug_1665745<Executor,Function,Factory,Factories...>::type::value,
                                use_strategy_15,
                                typename std::conditional<
                                  has_strategy_16<Executor,Function,Factory,Factories...>::value,
                                  use_strategy_16,
                                  typename std::conditional<
                                    has_strategy_17<Executor,Function,Factory,Factories...>::value,
                                    use_strategy_17,
                                    typename std::conditional<
                                      has_strategy_18<Executor,Function,Factory,Factories...>::value,
                                      use_strategy_18,
                                      typename std::conditional<
                                        has_strategy_19<Executor,Function,Factory,Factories...>::value,
                                        use_strategy_19,
                                        typename std::conditional<
                                          has_strategy_20<Executor,Function,Factory,Factories...>::value,
                                          use_strategy_20,
                                          typename std::conditional<
                                            has_strategy_21<Executor,Function,Factory,Factories...>::value,
                                            use_strategy_21,
                                            typename std::conditional<
                                              has_strategy_22<Executor,Function,Factory,Factories...>::value,
                                              use_strategy_22,
                                              typename std::conditional<
                                                has_strategy_23<Executor,Function,Factory,Factories...>::value,
                                                use_strategy_23,
                                                typename std::conditional<
                                                  has_strategy_24<Executor,Function,Factory,Factories...>::value,
                                                  use_strategy_24,
                                                  use_strategy_25
                                                >::type // 24
                                              >::type // 23
                                            >::type // 22
                                          >::type // 21
                                        >::type // 20
                                      >::type // 19
                                    >::type // 18
                                  >::type // 17
                                >::type // 16
                              >::type // 15
                            >::type // 14
                          >::type // 13
                        >::type // 12
                      >::type // 11
                    >::type // 10
                  >::type // 9
                >::type // 8
              >::type // 7
            >::type // 6
          >::type // 5
        >::type // 4
      >::type // 3
    >::type // 2
  >::type; // 1


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_execute_with_shared_inits_returning_user_specified_container_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  return ex.execute(f, result_factory, shape, shared_factories...);
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

  __agency_hd_warning_disable__
  template<size_t... ContainerIndices, class AgentIndex>
  __AGENCY_ANNOTATION
  Result impl(detail::index_sequence<ContainerIndices...>, AgentIndex&& agent_idx) const
  {
    return agency::invoke(
      f,
      std::forward<AgentIndex>(agent_idx),                                                             // pass the agent index
      std::get<ContainerIndices>(shared_arg_containers)[rank_in_group<ContainerIndices>(agent_idx)]... // pass the arguments coming in from shared parameters
    );
  }

  template<class Index>
  __AGENCY_ANNOTATION
  Result operator()(Index&& idx) const
  {
    constexpr size_t num_containers = std::tuple_size<TupleOfContainers>::value;
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


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_execute_returning_user_specified_container_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  // create a tuple of containers holding a shared parameter for each group
  auto shared_param_containers_tuple = make_tuple_of_shared_parameter_containers(ex, shape, shared_factories...);

  using index_type = typename new_executor_traits<Executor>::index_type;
  using result_type = typename std::result_of<
    Function(
      index_type,
      typename std::result_of<Factories()>::type&...
    )
  >::type;

  // wrap f with a functor to map container elements to shared parameters
  auto g = make_multi_agent_execute_with_shared_inits_functor<result_type>(f, shape, shared_param_containers_tuple);

  return ex.execute(g, result_factory, shape);
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


struct empty_type {};


template<class Result, class Container, class Function>
struct invoke_and_store_result_to_container
{
  Container& c;
  mutable Function f;

  __agency_hd_warning_disable__
  template<class Index, class... Args>
  __AGENCY_ANNOTATION
  Result operator()(const Index& idx, Args&... shared_args) const
  {
    c[idx] = agency::invoke(f, idx, shared_args...);

    // return something easily discardable
    return Result();
  }
};


__agency_hd_warning_disable__
template<class Result, class Container, class Function>
__AGENCY_ANNOTATION
invoke_and_store_result_to_container<Result,Container,Function> make_invoke_and_store_result_to_container(Container& c, Function f)
{
  return invoke_and_store_result_to_container<Result,Container,Function>{c,f};
} // end make_invoke_and_store_result_to_container()


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_execute_with_shared_inits_returning_void_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  auto results = result_factory(shape);

  ex.execute(make_invoke_and_store_result_to_container<void>(results,f), shape, shared_factories...);

  return results;
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_execute_returning_void_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  // create a tuple of containers holding a shared parameter for each group
  auto shared_param_containers_tuple = make_tuple_of_shared_parameter_containers(ex, shape, shared_factories...);

  using index_type = typename new_executor_traits<Executor>::index_type;
  using result_type = typename std::result_of<
    Function(
      index_type,
      typename std::result_of<Factories()>::type&...
    )
  >::type;

  // wrap f with a functor to map container elements to shared parameters
  auto g = make_multi_agent_execute_with_shared_inits_functor<result_type>(f, shape, shared_param_containers_tuple);

  // wrap g with a functor to store the result to a container
  auto results = result_factory(shape);
  auto h = make_invoke_and_store_result_to_container<void>(results, g);

  ex.execute(h, shape);

  return results;
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_async_execute_with_shared_inits_returning_user_specified_container_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  // XXX should go through executor_traits for the get()
  return ex.async_execute(f, result_factory, shape, shared_factories...).get();
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_then_execute_with_shared_inits_returning_user_specified_container_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  auto ready = new_executor_traits<Executor>::template make_ready_future<void>(ex);

  // XXX should go through executor_traits for the get()
  return ex.then_execute(f, result_factory, shape, ready, shared_factories...).get();
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Function>
struct strategy_7_functor
{
  mutable Function f;

  __agency_hd_warning_disable__
  template<class Index, class Container, class... Args>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Container& results, Args&... shared_args) const
  {
    results[idx] = agency::invoke(f, idx, shared_args...);
  }
};


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_when_all_execute_and_select_with_shared_inits_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  using container_type = typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type;
  auto results = new_executor_traits<Executor>::template make_ready_future<container_type>(ex, result_factory(shape));

  auto futures = detail::make_tuple(std::move(results));

  // XXX should go through executor_traits for the get()
  return ex.template when_all_execute_and_select<0>(strategy_7_functor<Function>{f}, shape, futures, shared_factories...).get();
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_async_execute_returning_user_specified_container_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  // create a tuple of containers holding a shared parameter for each group
  auto shared_param_containers_tuple = make_tuple_of_shared_parameter_containers(ex, shape, shared_factories...);

  using index_type = typename new_executor_traits<Executor>::index_type;
  using result_type = typename std::result_of<
    Function(
      index_type,
      typename std::result_of<Factories()>::type&...
    )
  >::type;

  // wrap f with a functor to map container elements to shared parameters
  auto g = make_multi_agent_execute_with_shared_inits_functor<result_type>(f, shape, shared_param_containers_tuple);

  // XXX should go through executor_traits for this get()
  return ex.async_execute(g, result_factory, shape).get();
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_then_execute_returning_user_specified_container_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  // create a tuple of containers holding a shared parameter for each group
  auto shared_param_containers_tuple = make_tuple_of_shared_parameter_containers(ex, shape, shared_factories...);

  using index_type = typename new_executor_traits<Executor>::index_type;
  using result_type = typename std::result_of<
    Function(
      index_type,
      typename std::result_of<Factories()>::type&...
    )
  >::type;

  // wrap f with a functor to map container elements to shared parameters
  auto g = make_multi_agent_execute_with_shared_inits_functor<result_type>(f, shape, shared_param_containers_tuple);

  auto ready = new_executor_traits<Executor>::template make_ready_future<void>(ex);

  // XXX should go through executor_traits for this get()
  return ex.then_execute(g, result_factory, shape, ready).get();
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Function>
struct invoke_and_store_to_second_parameter
{
  mutable Function f;

  __agency_hd_warning_disable__
  template<class Index, class Container>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Container& c) const
  {
    c[idx] = agency::invoke(f, idx);
  }
};

template<class Function>
__AGENCY_ANNOTATION
invoke_and_store_to_second_parameter<Function> make_invoke_and_store_to_second_parameter(Function f)
{
  return invoke_and_store_to_second_parameter<Function>{f};
}


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_when_all_execute_and_select_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  // create a tuple of containers holding a shared parameter for each group
  auto shared_param_containers_tuple = make_tuple_of_shared_parameter_containers(ex, shape, shared_factories...);

  using index_type = typename new_executor_traits<Executor>::index_type;
  using result_type = typename std::result_of<
    Function(
      index_type,
      typename std::result_of<Factories()>::type&...
    )
  >::type;

  // wrap f with a functor to map container elements to shared parameters
  auto g = make_multi_agent_execute_with_shared_inits_functor<result_type>(f, shape, shared_param_containers_tuple);

  // wrap g with a functor to store g's result to the container passed as the second parameter
  auto h = make_invoke_and_store_to_second_parameter(g);

  using container_type = typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type;
  auto results = new_executor_traits<Executor>::template make_ready_future<container_type>(ex, result_factory(shape));

  auto futures = detail::make_tuple(std::move(results));

  // XXX should go through executor_traits for the get()
  return ex.template when_all_execute_and_select<0>(h, shape, futures).get();
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_async_execute_with_shared_inits_returning_void_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  auto results = result_factory(shape);

  // XXX should call wait() through executor_traits
  ex.async_execute(make_invoke_and_store_result_to_container<void>(results, f), shape, shared_factories...).wait();

  return results;
}


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_then_execute_with_shared_inits_returning_void_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  auto results = result_factory(shape);

  auto ready = new_executor_traits<Executor>::template make_ready_future<void>(ex);

  // XXX should call wait() through executor_traits
  ex.then_execute(make_invoke_and_store_result_to_container<void>(results, f), shape, ready, shared_factories...).wait();

  return results;
}


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_execute_with_shared_inits_returning_default_container_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)

{
  auto results = result_factory(shape);

  // discard the container of results returned by this call
  ex.execute(make_invoke_and_store_result_to_container<empty_type>(results,f), shape, shared_factories...);

  return results;
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_async_execute_with_shared_inits_returning_default_container_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  auto results = result_factory(shape);

  // discard the container of results returned by this call
  // XXX should call wait() through executor_traits
  ex.async_execute(make_invoke_and_store_result_to_container<empty_type>(results,f), shape, shared_factories...).wait();

  return results;
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_async_execute_returning_void_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  // create a tuple of containers holding a shared parameter for each group
  auto shared_param_containers_tuple = make_tuple_of_shared_parameter_containers(ex, shape, shared_factories...);

  using index_type = typename new_executor_traits<Executor>::index_type;
  using result_type = typename std::result_of<
    Function(
      index_type,
      typename std::result_of<Factories()>::type&...
    )
  >::type;

  // wrap f with a functor to map container elements to shared parameters
  auto g = make_multi_agent_execute_with_shared_inits_functor<result_type>(f, shape, shared_param_containers_tuple);

  auto results = result_factory(shape);

  // wrap g with a functor to store g's results to a container
  auto h = make_invoke_and_store_result_to_container<void>(results, g);

  // discard the container returned by this call
  // XXX wait() should be called through executor_traits
  ex.async_execute(h, shape).wait();

  return results;
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_then_execute_returning_void_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  // create a tuple of containers holding a shared parameter for each group
  auto shared_param_containers_tuple = make_tuple_of_shared_parameter_containers(ex, shape, shared_factories...);

  using index_type = typename new_executor_traits<Executor>::index_type;
  using result_type = typename std::result_of<
    Function(
      index_type,
      typename std::result_of<Factories()>::type&...
    )
  >::type;

  // wrap f with a functor to map container elements to shared parameters
  auto g = make_multi_agent_execute_with_shared_inits_functor<result_type>(f, shape, shared_param_containers_tuple);

  auto results = result_factory(shape);

  // wrap g with a functor to store g's results to a container
  auto h = make_invoke_and_store_result_to_container<void>(results, g);

  auto ready = new_executor_traits<Executor>::template make_ready_future<void>(ex);

  // discard the container returned by this call
  // XXX wait() should be called through executor_traits
  ex.then_execute(h, shape, ready).wait();

  return results;
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_then_execute_with_shared_inits_returning_default_container_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  auto results = result_factory(shape);

  auto ready = new_executor_traits<Executor>::template make_ready_future<void>(ex);

  // discard the container of results returned by this call
  // XXX should call wait() through executor_traits
  ex.then_execute(make_invoke_and_store_result_to_container<empty_type>(results,f), shape, ready, shared_factories...).wait();

  return results;
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_execute_returning_default_container_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  // create a tuple of containers holding a shared parameter for each group
  auto shared_param_containers_tuple = make_tuple_of_shared_parameter_containers(ex, shape, shared_factories...);

  using index_type = typename new_executor_traits<Executor>::index_type;
  using result_type = typename std::result_of<
    Function(
      index_type,
      typename std::result_of<Factories()>::type&...
    )
  >::type;

  // wrap f with a functor to map container elements to shared parameters
  auto g = make_multi_agent_execute_with_shared_inits_functor<result_type>(f, shape, shared_param_containers_tuple);

  auto results = result_factory(shape);

  // wrap g with a functor to store g's results to a container
  auto h = make_invoke_and_store_result_to_container<empty_type>(results, g);

  // discard the container returned by this call
  ex.execute(h, shape);

  return results;
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_async_execute_returning_default_container_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  // create a tuple of containers holding a shared parameter for each group
  auto shared_param_containers_tuple = make_tuple_of_shared_parameter_containers(ex, shape, shared_factories...);

  using index_type = typename new_executor_traits<Executor>::index_type;
  using result_type = typename std::result_of<
    Function(
      index_type,
      typename std::result_of<Factories()>::type&...
    )
  >::type;

  // wrap f with a functor to map container elements to shared parameters
  auto g = make_multi_agent_execute_with_shared_inits_functor<result_type>(f, shape, shared_param_containers_tuple);

  auto results = result_factory(shape);

  // wrap g with a functor to store g's results to a container
  auto h = make_invoke_and_store_result_to_container<empty_type>(results, g);

  // discard the container returned by this call
  // XXX should call wait() through executor_traits
  ex.async_execute(h, shape).wait();

  return results;
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_then_execute_returning_default_container_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  // create a tuple of containers holding a shared parameter for each group
  auto shared_param_containers_tuple = make_tuple_of_shared_parameter_containers(ex, shape, shared_factories...);

  using index_type = typename new_executor_traits<Executor>::index_type;
  using result_type = typename std::result_of<
    Function(
      index_type,
      typename std::result_of<Factories()>::type&...
    )
  >::type;

  // wrap f with a functor to map container elements to shared parameters
  auto g = make_multi_agent_execute_with_shared_inits_functor<result_type>(f, shape, shared_param_containers_tuple);

  auto results = result_factory(shape);

  // wrap g with a functor to store g's results to a container
  auto h = make_invoke_and_store_result_to_container<empty_type>(results, g);

  auto ready = new_executor_traits<Executor>::template make_ready_future<void>(ex);

  // discard the container returned by this call
  // XXX should call wait() through executor_traits
  ex.then_execute(h, shape, ready).wait();

  return results;
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Function, class Factory1, class Shape, class Factory2>
struct execute_in_for_loop
{
  mutable Function f;
  mutable Factory1 result_factory;
  Shape shape;
  mutable Factory2 shared_factory;

  __agency_hd_warning_disable__
  template<class F, class F1, class S, class F2>
  __AGENCY_ANNOTATION
  execute_in_for_loop(F&& f_, F1&& result_factory_, S&& shape_, F2&& shared_factory_)
    : f(std::forward<F>(f_)),
      result_factory(std::forward<F1>(result_factory_)),
      shape(std::forward<S>(shape_)),
      shared_factory(std::forward<F2>(shared_factory_))
  {}

  __agency_hd_warning_disable__
  __AGENCY_ANNOTATION
  typename std::result_of<Factory1(Shape)>::type
    operator()() const
  {
    auto results = result_factory(shape);
    auto shared_arg = shared_factory();

    // XXX generalize to multidimensions
    for(size_t idx = 0; idx < shape; ++idx)
    {
      results[idx] = agency::invoke(f, idx, shared_arg);
    }

    return results;
  }
};

template<class Function, class Factory1, class Shape, class Factory2>
__AGENCY_ANNOTATION
execute_in_for_loop<Function,Factory1,Shape,Factory2> make_execute_in_for_loop(Function f, Factory1 result_factory, Shape shape, Factory2 shared_factory)
{
  return execute_in_for_loop<Function,Factory1,Shape,Factory2>(f, result_factory, shape, shared_factory);
}


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_single_agent_execute_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  static_assert(sizeof...(Factories) == 1, "This implementation only makes sense for flat (i.e., single-agent) executors");

  return ex.execute(make_execute_in_for_loop(f, result_factory, shape, shared_factories...));
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_single_agent_async_execute_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  static_assert(sizeof...(Factories) == 1, "This implementation only makes sense for flat (i.e., single-agent) executors");

  // XXX should call get() through executor_traits
  return ex.async_execute(make_execute_in_for_loop(f, result_factory, shape, shared_factories...)).get();
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_single_agent_then_execute_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  static_assert(sizeof...(Factories) == 1, "This implementation only makes sense for flat (i.e., single-agent) executors");

  auto ready = new_executor_traits<Executor>::template make_ready_future<void>(ex);

  // XXX should call get() through executor_traits
  return ex.then_execute(make_execute_in_for_loop(f, result_factory, shape, shared_factories...), ready).get();
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Function, class Shape, class Factory>
struct single_agent_when_all_execute_and_select_functor
{
  mutable Function f;
  Shape shape;
  mutable Factory shared_factory;

  __agency_hd_warning_disable__
  template<class Container>
  __AGENCY_ANNOTATION
  void operator()(Container& results) const
  {
    auto shared_arg = shared_factory();

    // XXX generalize to multidimensions
    for(size_t idx = 0; idx < shape; ++idx)
    {
      results[idx] = agency::invoke(f, idx, shared_arg);
    }
  }
};


template<class Function, class Shape, class Factory>
__AGENCY_ANNOTATION
single_agent_when_all_execute_and_select_functor<Function,Shape,Factory> make_single_agent_when_all_execute_and_select_functor(Function f, Shape shape, Factory shared_factory)
{
  return single_agent_when_all_execute_and_select_functor<Function,Shape,Factory>{f, shape, shared_factory};
}


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_single_agent_when_all_execute_and_select_member_function,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  static_assert(sizeof...(Factories) == 1, "This implementation only makes sense for flat (i.e., single-agent) executors");

  using result_type = typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type;

  auto results = new_executor_traits<Executor>::template make_ready_future<result_type>(ex, result_factory(shape));
  auto tuple_of_futures = detail::make_tuple(std::move(results));

  // XXX should call get() through executor_traits
  return ex.template when_all_execute_and_select<0>(make_single_agent_when_all_execute_and_select_functor(f, shape, shared_factories...), tuple_of_futures).get();
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Executor, class Function, class Factory, class... Factories>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
  multi_agent_execute_with_shared_inits_returning_user_specified_container(use_bare_for_loop,
                                                                           Executor& ex,
                                                                           Function f,
                                                                           Factory result_factory,
                                                                           typename new_executor_traits<Executor>::shape_type shape,
                                                                           Factories... shared_factories)
{
  static_assert(sizeof...(Factories) == 1, "This implementation only makes sense for flat (i.e., single-agent) executors");

  auto implementation = make_execute_in_for_loop(f, result_factory, shape, shared_factories...);
  return implementation();
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


} // end new_multi_agent_execute_with_shared_inits_returning_user_specified_container_implementation_strategies
} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Function, class Factory, class... Factories,
           class Enable>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type new_executor_traits<Executor>
  ::execute(typename new_executor_traits<Executor>::executor_type& ex,
            Function f,
            Factory result_factory,
            typename new_executor_traits<Executor>::shape_type shape,
            Factories... shared_factories)
{
  namespace ns = detail::new_executor_traits_detail::new_multi_agent_execute_with_shared_inits_returning_user_specified_container_implementation_strategies;

  using implementation_strategy = ns::select_multi_agent_execute_with_shared_inits_returning_user_specified_container_implementation<
    Executor,
    Function,
    Factory,
    Factories...
  >;

  return ns::multi_agent_execute_with_shared_inits_returning_user_specified_container(implementation_strategy(), ex, f, result_factory, shape, shared_factories...);
} // end new_executor_traits::execute()


} // end agency

