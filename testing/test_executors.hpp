#pragma once

#include <agency/future.hpp>


struct empty_executor
{
  bool valid() { return true; }
};


struct simple_single_agent_when_all_execute_and_select_executor
{
  simple_single_agent_when_all_execute_and_select_executor() : function_called{} {};

  template<size_t... SelectedIndices, class Function, class TupleOfFutures>
  std::future<
    agency::detail::when_all_execute_and_select_result_t<
      agency::detail::index_sequence<SelectedIndices...>,
      typename std::decay<TupleOfFutures>::type
    >
  >
    when_all_execute_and_select(Function f, TupleOfFutures&& futures)
  {
    function_called = true;

    return agency::when_all_execute_and_select<SelectedIndices...>(f, std::forward<TupleOfFutures>(futures));
  }

  bool function_called;

  bool valid()
  {
    return function_called;
  }
};


struct simple_multi_agent_when_all_execute_and_select_executor
{
  simple_multi_agent_when_all_execute_and_select_executor() : function_called{} {};

  template<class Function>
  struct functor
  {
    mutable Function f;
    size_t n;

    template<class... Args>
    void operator()(Args&&... args) const
    {
      for(size_t idx = 0; idx < n; ++idx)
      {
        f(idx, std::forward<Args>(args)...);
      }
    }
  };

  template<size_t... SelectedIndices, class Function, class TupleOfFutures>
  std::future<
    agency::detail::when_all_execute_and_select_result_t<
      agency::detail::index_sequence<SelectedIndices...>,
      typename std::decay<TupleOfFutures>::type
    >
  >
    when_all_execute_and_select(Function f, size_t n, TupleOfFutures&& futures)
  {
    function_called = true;

    return agency::when_all_execute_and_select<SelectedIndices...>(functor<Function>{f, n}, std::forward<TupleOfFutures>(futures));
  }

  bool function_called;

  bool valid()
  {
    return function_called;
  }
};


struct simple_multi_agent_when_all_execute_and_select_with_shared_inits_executor
{
  simple_multi_agent_when_all_execute_and_select_with_shared_inits_executor() : function_called{} {};

  template<class Function, class T>
  struct functor
  {
    mutable Function f;
    size_t n;
    T shared_arg;

    template<class... Args>
    void operator()(Args&&... args) const
    {
      for(size_t idx = 0; idx < n; ++idx)
      {
        f(idx, std::forward<Args>(args)..., shared_arg);
      }
    }
  };

  template<size_t... SelectedIndices, class Function, class TupleOfFutures, class T>
  std::future<
    agency::detail::when_all_execute_and_select_result_t<
      agency::detail::index_sequence<SelectedIndices...>,
      typename std::decay<TupleOfFutures>::type
    >
  >
    when_all_execute_and_select(Function f, size_t n, TupleOfFutures&& futures, T&& shared_init)
  {
    function_called = true;

    auto g = functor<Function, typename std::decay<T>::type>{f, n, std::forward<T>(shared_init)};
    return agency::when_all_execute_and_select<SelectedIndices...>(std::move(g), std::forward<TupleOfFutures>(futures));
  }

  bool function_called;

  bool valid()
  {
    return function_called;
  }
};

struct simple_single_agent_then_execute_executor
{
  simple_single_agent_then_execute_executor() : function_called{} {};

  template<class Function, class T>
  std::future<typename std::result_of<Function(T&)>::type>
    then_execute(Function f, std::future<T>& fut)
  {
    function_called = true;

    return agency::detail::then(fut, [=](std::future<T>& fut)
    {
      auto arg = fut.get();
      return f(arg);
    });
  }

  template<class Function>
  std::future<typename std::result_of<Function()>::type>
    then_execute(Function f, std::future<void>& fut)
  {
    function_called = true;

    return agency::detail::then(fut, [=](std::future<void>& fut)
    {
      return f();
    });
  }

  bool function_called;

  bool valid()
  {
    return function_called;
  }
};

