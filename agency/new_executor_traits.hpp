#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/execution_categories.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/executor_traits/member_types.hpp>
#include <vector>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{



template<class Executor, class Function, class TupleOfFutures>
using has_single_agent_when_all_execute = typename has_single_agent_when_all_execute_impl<Executor, Function, TupleOfFutures>::type;


} // end new_executor_traits_detail
} // end detail


template<class Executor>
struct new_executor_traits
{
  public:
    using executor_type = Executor;

    using execution_category = detail::new_executor_traits_detail::executor_execution_category_t<executor_type>;

    constexpr static size_t execution_depth = detail::execution_depth<execution_category>::value;

    using index_type = detail::new_executor_traits_detail::executor_index_t<executor_type>;

    using shape_type = typename detail::new_executor_traits_detail::nested_shape_type_with_default<
      executor_type,
      index_type
    >::type;

    template<class T>
    using future = typename detail::new_executor_traits_detail::executor_future_t<executor_type,T>;

    template<class T>
    using shared_future = typename detail::new_executor_traits_detail::executor_shared_future_t<executor_type,T>;

    //template<class T>
    //using container = typename detail::new_executor_traits_detail::nested_container_with_default<
    //  executor_type,
    //  std::vector
    //>::template type_template<T>;
    template<class T>
    using container = detail::new_executor_traits_detail::member_container_or_t<std::vector<T>, executor_type, T>;

    // XXX this should be the other way around - container<T> should depend on allocator<T>
    // XXX should check the executor for the allocator
    template<class T>
    using allocator = typename container<T>::allocator_type;

    template<class T, class... Args>
    __AGENCY_ANNOTATION
    static future<T> make_ready_future(executor_type& ex, Args&&... args);

    template<class T, class Future>
    __AGENCY_ANNOTATION
    static future<T> future_cast(executor_type& ex, Future& fut);

    template<class Future>
    __AGENCY_ANNOTATION
    static shared_future<typename future_traits<Future>::value_type>
      share_future(executor_type& ex, Future& fut);

    template<class Future, class Factory>
    __AGENCY_ANNOTATION
    static typename std::result_of<Factory(shape_type)>::type
      share_future(executor_type& ex, Future& fut, Factory result_factory, shape_type shape);

    template<class Future>
    __AGENCY_ANNOTATION
    static container<shared_future<typename future_traits<Future>::value_type>>
      share_future(executor_type& ex, Future& fut, shape_type shape);

    template<class... Futures>
    __AGENCY_ANNOTATION
    static future<
      detail::when_all_result_t<typename std::decay<Futures>::type...>
    > when_all(executor_type& ex, Futures&&... futures);

    // single-agent when_all_execute_and_select()
    template<size_t... Indices, class Function, class TupleOfFutures>
    static future<
      detail::when_all_execute_and_select_result_t<
        detail::index_sequence<Indices...>,
        typename std::decay<TupleOfFutures>::type
      >
    >
      when_all_execute_and_select(executor_type& ex, Function&& f, TupleOfFutures&& futures);

    // multi-agent when_all_execute_and_select()
    template<size_t... Indices, class Function, class TupleOfFutures>
    static future<
      detail::when_all_execute_and_select_result_t<
        detail::index_sequence<Indices...>,
        typename std::decay<TupleOfFutures>::type
      >
    >
      when_all_execute_and_select(executor_type& ex, Function f, shape_type shape, TupleOfFutures&& futures);

    // multi-agent when_all_execute_and_select() with shared parameters
    template<size_t... Indices, class Function, class TupleOfFutures, class Factory, class... Factories>
    static future<
      detail::when_all_execute_and_select_result_t<
        detail::index_sequence<Indices...>,
        typename std::decay<TupleOfFutures>::type
      >
    >
      when_all_execute_and_select(executor_type& ex, Function f, shape_type shape, TupleOfFutures&& futures, Factory outer_shared_factory, Factories... inner_shared_factories);

    // single-agent then_execute()
    template<class Function, class Future>
    __AGENCY_ANNOTATION
    static future<
      detail::result_of_continuation_t<Function,Future>
    >
      then_execute(executor_type& ex, Function&& f, Future& fut);

    // multi-agent then_execute() returning user-specified Container
    template<class Function, class Future, class Factory,
             class = typename std::enable_if<
               detail::is_future<Future>::value
             >::type,
             class = detail::result_of_continuation_t<
               Function,
               index_type,
               Future
             >
            >
    __AGENCY_ANNOTATION
    static future<typename std::result_of<Factory(shape_type)>::type> then_execute(executor_type& ex, Function f, Factory result_factory, shape_type shape, Future& fut);

    // multi-agent then_execute() with shared inits returning user-specified Container
    template<class Function, class Factory, class Future, class... Factories,
             class = typename std::enable_if<
               detail::is_future<Future>::value
             >::type,
             class = typename std::enable_if<
               sizeof...(Factories) == execution_depth
             >::type,
             class = detail::result_of_continuation_t<
               Function,
               index_type,
               Future,
               typename std::result_of<Factories()>::type&...
             >>
    __AGENCY_ANNOTATION
    static future<typename std::result_of<Factory(shape_type)>::type> then_execute(executor_type& ex, Function f, Factory result_factory, shape_type shape, Future& fut, Factories... shared_factories);

    // multi-agent then_execute() returning default container
    template<class Function, class Future,
             class = typename std::enable_if<
               detail::is_future<Future>::value
             >::type,
             class = typename std::enable_if<
               detail::is_callable_continuation<Function,index_type,Future>::value
             >::type,
             class = typename std::enable_if<
               !std::is_void<
                 detail::result_of_continuation_t<Function,index_type,Future>
               >::value
             >::type>
    static future<
      container<
        detail::result_of_continuation_t<Function,index_type,Future>
      >
    >
      then_execute(executor_type& ex, Function f, shape_type shape, Future& fut);

    // multi-agent then_execute() with shared inits returning default container
    template<class Function, class Future, class... Factories,
             class = typename std::enable_if<
               detail::is_future<Future>::value
             >::type,
             class = typename std::enable_if<
               sizeof...(Factories) == execution_depth
             >::type,
             class = typename std::enable_if<
               detail::is_callable_continuation<Function,index_type,Future,typename std::result_of<Factories()>::type&...>::value
             >::type,
             class = typename std::enable_if<
               !std::is_void<
                 detail::result_of_continuation_t<Function,index_type,Future,typename std::result_of<Factories()>::type&...>
               >::value
             >::type>
    static future<
      container<
        detail::result_of_continuation_t<Function,index_type,Future,typename std::result_of<Factories()>::type&...>
      >
    >
      then_execute(executor_type& ex, Function f, shape_type shape, Future& fut, Factories... shared_factories);

    // multi-agent then_execute() returning void
    template<class Function, class Future,
             class = typename std::enable_if<
               detail::is_future<Future>::value
             >::type,
             class = typename std::enable_if<
               detail::is_callable_continuation<Function,index_type,Future>::value
             >::type,
             class = typename std::enable_if<
               std::is_void<
                 detail::result_of_continuation_t<Function,index_type,Future>
               >::value
             >::type>
    __AGENCY_ANNOTATION
    static future<void>
      then_execute(executor_type& ex, Function f, shape_type shape, Future& fut);

    // multi-agent then_execute() with shared inits returning void
    template<class Function, class Future, class... Factories,
             class = typename std::enable_if<
               detail::is_future<Future>::value
             >::type,
             class = typename std::enable_if<
               sizeof...(Factories) == execution_depth
             >::type,
             class = typename std::enable_if<
               detail::is_callable_continuation<Function,index_type,Future,typename std::result_of<Factories()>::type&...>::value
             >::type,
             class = typename std::enable_if<
               std::is_void<
                 detail::result_of_continuation_t<Function,index_type,Future,typename std::result_of<Factories()>::type&...>
               >::value
             >::type>
    __AGENCY_ANNOTATION
    static future<void>
      then_execute(executor_type& ex, Function f, shape_type shape, Future& fut, Factories... shared_factories);

    // single-agent async_execute()
    template<class Function>
    static future<
      typename std::result_of<Function()>::type
    >
      async_execute(executor_type& ex, Function&& f);

    // multi-agent async_execute() returning user-specified Container
    template<class Function, class Factory>
    static future<typename std::result_of<Factory(shape_type)>::type> async_execute(executor_type& ex, Function f, Factory result_factory, shape_type shape);

    // multi-agent async_execute() with shared inits returning user-specified Container
    template<class Function, class Factory, class... Factories,
             class = typename std::enable_if<
               sizeof...(Factories) == execution_depth
             >::type>
    __AGENCY_ANNOTATION
    static future<typename std::result_of<Factory(shape_type)>::type> async_execute(executor_type& ex, Function f, Factory result_factory, shape_type shape, Factories... shared_factories);

    // multi-agent async_execute() returning default container
    template<class Function,
             class = typename std::enable_if<
               !std::is_void<
                typename std::result_of<Function(index_type)>::type
               >::value
             >::type>
    static future<
      container<
        typename std::result_of<Function(index_type)>::type
      >
    >
      async_execute(executor_type& ex, Function f, shape_type shape);

    // multi-agent async_execute() with shared inits returning default container
    template<class Function,
             class... Factories,
             class = typename std::enable_if<
               sizeof...(Factories) == execution_depth
             >::type,
             class = typename std::enable_if<
               !std::is_void<
                 typename std::result_of<Function(index_type, typename std::result_of<Factories()>::type&...)>::type
               >::value
             >::type>
    static future<
      container<
        typename std::result_of<Function(index_type, typename std::result_of<Factories()>::type&...)>::type
      >
    >
      async_execute(executor_type& ex, Function f, shape_type shape, Factories... shared_factories);

    // multi-agent async_execute() returning void
    template<class Function,
             class = typename std::enable_if<
              std::is_void<
                typename std::result_of<Function(index_type)>::type
              >::value
             >::type>
    static future<void> async_execute(executor_type& ex, Function f, shape_type shape);

    // multi-agent async_execute() with shared inits returning void
    template<class Function,
             class... Factories,
             class = typename std::enable_if<
               sizeof...(Factories) == execution_depth
             >::type,
             class = typename std::enable_if<
               std::is_void<
                 typename std::result_of<Function(index_type, typename std::result_of<Factories()>::type&...)>::type
               >::value
             >::type>
    __AGENCY_ANNOTATION
    static future<void> async_execute(executor_type& ex, Function f, shape_type shape, Factories... shared_factories);

    // single-agent execute()
    template<class Function>
    __AGENCY_ANNOTATION
    static typename std::result_of<Function()>::type
      execute(executor_type& ex, Function&& f);

    // multi-agent execute returning user-specified Container
    template<class Function, class Factory>
    __AGENCY_ANNOTATION
    static typename std::result_of<Factory(shape_type)>::type execute(executor_type& ex, Function f, Factory result_factory, shape_type shape);

    // multi-agent execute with shared inits returning user-specified Container
    template<class Function, class Factory, class... Factories,
             class = typename std::enable_if<
               execution_depth == sizeof...(Factories)
             >::type>
    __AGENCY_ANNOTATION
    static typename std::result_of<Factory(shape_type)>::type execute(executor_type& ex, Function f, Factory result_factory, shape_type shape, Factories... shared_factories);

    // multi-agent execute returning default container
    template<class Function,
             class = typename std::enable_if<
               !std::is_void<
                 typename std::result_of<
                   Function(index_type)
                 >::type
               >::value
             >::type>
    static container<
      typename std::result_of<Function(index_type)>::type
    >
      execute(executor_type& ex, Function f, shape_type shape);

    // multi-agent execute with shared inits returning default container
    template<class Function, class... Factories,
             class = typename std::enable_if<
               !std::is_void<
                 typename std::result_of<
                   Function(index_type, typename std::result_of<Factories()>::type&...)
                 >::type
               >::value
             >::type,
             class = typename std::enable_if<
               execution_depth == sizeof...(Factories)
             >::type>
    static container<
      typename std::result_of<
        Function(
          index_type,
          typename std::result_of<Factories()>::type&...
        )
      >::type
    >
      execute(executor_type& ex, Function f, shape_type shape, Factories... shared_factories);

    // multi-agent execute returning void
    template<class Function,
             class = typename std::enable_if<
               std::is_void<
                 typename std::result_of<
                   Function(index_type)
                 >::type
               >::value
             >::type>
    __AGENCY_ANNOTATION
    static void execute(executor_type& ex, Function f, shape_type shape);

    // multi-agent execute with shared inits returning void
    template<class Function, class... Factories,
             class = typename std::enable_if<
               std::is_void<
                 typename std::result_of<
                   Function(index_type, typename std::result_of<Factories()>::type&...)
                 >::type
               >::value
             >::type,
             class = typename std::enable_if<
               execution_depth == sizeof...(Factories)
             >::type>
    __AGENCY_ANNOTATION
    static void execute(executor_type& ex, Function f, shape_type shape, Factories... shared_factories);
}; // end new_executor_traits


} // end agency

#include <agency/detail/executor_traits/make_ready_future.hpp>
#include <agency/detail/executor_traits/future_cast.hpp>
#include <agency/detail/executor_traits/share_future.hpp>
#include <agency/detail/executor_traits/single_agent_when_all_execute_and_select.hpp>
#include <agency/detail/executor_traits/multi_agent_when_all_execute_and_select.hpp>
#include <agency/detail/executor_traits/multi_agent_when_all_execute_and_select_with_shared_inits.hpp>
#include <agency/detail/executor_traits/single_agent_then_execute.hpp>
#include <agency/detail/executor_traits/multi_agent_then_execute_returning_user_specified_container.hpp>
#include <agency/detail/executor_traits/multi_agent_then_execute_returning_default_container.hpp>
#include <agency/detail/executor_traits/multi_agent_then_execute_returning_void.hpp>
#include <agency/detail/executor_traits/multi_agent_then_execute_with_shared_inits_returning_user_specified_container.hpp>
#include <agency/detail/executor_traits/multi_agent_then_execute_with_shared_inits_returning_default_container.hpp>
#include <agency/detail/executor_traits/multi_agent_then_execute_with_shared_inits_returning_void.hpp>
#include <agency/detail/executor_traits/single_agent_async_execute.hpp>
#include <agency/detail/executor_traits/multi_agent_async_execute_returning_user_specified_container.hpp>
#include <agency/detail/executor_traits/multi_agent_async_execute_returning_default_container.hpp>
#include <agency/detail/executor_traits/multi_agent_async_execute_returning_void.hpp>
#include <agency/detail/executor_traits/multi_agent_async_execute_with_shared_inits_returning_user_specified_container.hpp>
#include <agency/detail/executor_traits/multi_agent_async_execute_with_shared_inits_returning_default_container.hpp>
#include <agency/detail/executor_traits/multi_agent_async_execute_with_shared_inits_returning_void.hpp>
#include <agency/detail/executor_traits/single_agent_execute.hpp>
#include <agency/detail/executor_traits/multi_agent_execute_returning_user_specified_container.hpp>
#include <agency/detail/executor_traits/multi_agent_execute_returning_default_container.hpp>
#include <agency/detail/executor_traits/multi_agent_execute_returning_void.hpp>
#include <agency/detail/executor_traits/multi_agent_execute_with_shared_inits_returning_user_specified_container.hpp>
#include <agency/detail/executor_traits/multi_agent_execute_with_shared_inits_returning_default_container.hpp>
#include <agency/detail/executor_traits/multi_agent_execute_with_shared_inits_returning_void.hpp>
#include <agency/detail/executor_traits/when_all.hpp>

