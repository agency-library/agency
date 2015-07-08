#pragma once

#include <agency/future.hpp>

namespace test_executors
{


struct test_executor
{
  bool function_called;

  test_executor() : function_called(false) {};

  bool valid() { return function_called; }
};


struct empty_executor : test_executor
{
  empty_executor()
  {
    function_called = true;
  }
};


struct single_agent_execute_executor : test_executor
{
  template<class Function>
  typename std::result_of<Function()>::type execute(Function&& f)
  {
    function_called = true;

    return std::forward<Function>(f)();
  }
};


struct single_agent_async_execute_executor : test_executor
{
  template<class Function>
  std::future<
    typename std::result_of<Function()>::type
  >
    async_execute(Function&& f)
  {
    function_called = true;

    return std::async(std::forward<Function>(f));
  }
};


struct single_agent_then_execute_executor : test_executor
{
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
};


struct single_agent_when_all_execute_and_select_executor : test_executor
{
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
};


struct multi_agent_when_all_execute_and_select_executor : test_executor
{
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
};


struct multi_agent_when_all_execute_and_select_with_shared_inits_executor : test_executor
{
  template<class Function, class T>
  struct functor
  {
    mutable Function f;
    size_t n;
    mutable T shared_arg;

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
};


struct when_all_executor : test_executor
{
  template<class... Futures>
  std::future<
    agency::detail::when_all_result_t<
      typename std::decay<Futures>::type...
    >
  >
    when_all(Futures&&... futures)
  {
    function_called = true;

    return agency::when_all(std::forward<Futures>(futures)...);
  }
};


struct multi_agent_execute_returning_user_defined_container_executor : test_executor
{
  template<class Container, class Function>
  Container execute(Function f, size_t n)
  {
    function_called = true;

    Container result(n);

    for(size_t i = 0; i < n; ++i)
    {
      result[i] = f(i);
    }

    return result;
  }
};


struct multi_agent_execute_with_shared_inits_returning_user_defined_container_executor : test_executor
{
  template<class Container, class Function, class T>
  Container execute(Function f, size_t n, T&& shared_init)
  {
    function_called = true;

    Container result(n);

    auto shared_arg = std::forward<T>(shared_init);

    for(size_t i = 0; i < n; ++i)
    {
      result[i] = f(i, shared_arg);
    }

    return result;
  }
};


struct multi_agent_execute_returning_default_container_executor : test_executor
{
  template<class Function>
  std::vector<
    typename std::result_of<Function(size_t)>::type
  > execute(Function f, size_t n)
  {
    function_called = true;

    std::vector<typename std::result_of<Function(size_t)>::type> result(n);

    for(size_t i = 0; i < n; ++i)
    {
      result[i] = f(i);
    }

    return result;
  }
};


struct multi_agent_execute_with_shared_inits_returning_default_container_executor : test_executor
{
  template<class Function, class T>
  std::vector<
    typename std::result_of<Function(size_t, typename std::decay<T>::type&)>::type
  > execute(Function f, size_t n, T&& shared_init)
  {
    function_called = true;

    std::vector<typename std::result_of<Function(size_t, typename std::decay<T>::type&)>::type> result(n);

    auto shared_arg = std::forward<T>(shared_init);

    for(size_t i = 0; i < n; ++i)
    {
      result[i] = f(i, shared_arg);
    }

    return result;
  }
};


struct multi_agent_execute_returning_void_executor : test_executor
{
  template<class Function>
  void execute(Function f, size_t n)
  {
    function_called = true;

    for(size_t i = 0; i < n; ++i)
    {
      f(i);
    }
  }
};


struct multi_agent_execute_with_shared_inits_returning_void_executor : test_executor
{
  template<class Function, class T>
  void execute(Function f, size_t n, T&& shared_init)
  {
    function_called = true;

    auto shared_arg = std::forward<T>(shared_init);

    for(size_t i = 0; i < n; ++i)
    {
      f(i, shared_arg);
    }
  }
};


struct multi_agent_async_execute_returning_user_defined_container_executor : test_executor
{
  template<class Container, class Function>
  std::future<Container> async_execute(Function f, size_t n)
  {
    function_called = true;

    return std::async([=]
    {
      multi_agent_execute_returning_user_defined_container_executor exec;

      return exec.execute<Container>(f, n);
    });
  }
};


struct multi_agent_async_execute_with_shared_inits_returning_user_defined_container_executor : test_executor
{
  template<class Container, class Function, class T>
  std::future<Container> async_execute(Function f, size_t n, T&& shared_init)
  {
    function_called = true;

    return std::async([=]
    {
      multi_agent_execute_with_shared_inits_returning_user_defined_container_executor exec;

      return exec.execute<Container>(f, n, shared_init);
    });
  }
};


struct multi_agent_async_execute_returning_default_container_executor : test_executor
{
  template<class Function>
  std::future<
    std::vector<
      typename std::result_of<Function(size_t)>::type
    >
  > async_execute(Function f, size_t n)
  {
    function_called = true;

    multi_agent_async_execute_returning_user_defined_container_executor exec;

    using container_type = std::vector<
      typename std::result_of<Function(size_t)>::type
    >;

    return exec.async_execute<container_type>(f, n);
  }
};


struct multi_agent_async_execute_with_shared_inits_returning_default_container_executor : test_executor
{
  template<class Function, class T>
  std::future<
    std::vector<
      typename std::result_of<Function(size_t, typename std::decay<T>::type&)>::type
    >
  > async_execute(Function f, size_t n, T&& shared_init)
  {
    function_called = true;

    multi_agent_async_execute_with_shared_inits_returning_user_defined_container_executor exec;

    using container_type = std::vector<
      typename std::result_of<Function(size_t, typename std::decay<T>::type&)>::type
    >;

    return exec.async_execute<container_type>(f, n, std::forward<T>(shared_init));
  }
};


struct multi_agent_async_execute_returning_void_executor : test_executor
{
  template<class Function>
  std::future<void> async_execute(Function f, size_t n)
  {
    function_called = true;

    return std::async([=]
    {
      multi_agent_execute_returning_void_executor exec;

      exec.execute(f, n);
    });
  }
};


struct multi_agent_async_execute_with_shared_inits_returning_void_executor : test_executor
{
  template<class Function, class T>
  std::future<void> async_execute(Function f, size_t n, T&& shared_init)
  {
    function_called = true;

    return std::async([=]
    {
      multi_agent_execute_with_shared_inits_returning_void_executor exec;

      exec.execute(f, n, shared_init);
    });
  }
};


struct multi_agent_then_execute_returning_user_defined_container_executor : test_executor
{
  template<class Container, class Function, class T>
  std::future<Container> then_execute(Function f, size_t n, std::future<T>& fut)
  {
    function_called = true;

    T val = fut.get(); 

    multi_agent_async_execute_returning_user_defined_container_executor exec;

    // we need mutable on the lambda because we pass val to f via mutable reference
    return exec.async_execute<Container>([=](const size_t& idx) mutable
    {
      return f(idx, val);
    },
    n);
  }

  template<class Container, class Function>
  std::future<Container> then_execute(Function f, size_t n, std::future<void>& fut)
  {
    function_called = true;

    fut.get(); 

    multi_agent_async_execute_returning_user_defined_container_executor exec;

    return exec.async_execute<Container>(f, n);
  }
};


struct multi_agent_then_execute_returning_default_container_executor : test_executor
{
  template<class Function, class T>
  std::future<
    std::vector<
      typename std::result_of<Function(size_t,T&)>::type
    >
  >
    then_execute(Function f, size_t n, std::future<T>& fut)
  {
    function_called = true;

    multi_agent_then_execute_returning_user_defined_container_executor exec;

    using container_type = std::vector<
      typename std::result_of<Function(size_t,T&)>::type
    >;

    return exec.then_execute<container_type>(f, n, fut);
  }

  template<class Function>
  std::future<
    std::vector<
      typename std::result_of<Function(size_t)>::type
    >
  >
    then_execute(Function f, size_t n, std::future<void>& fut)
  {
    function_called = true;

    multi_agent_then_execute_returning_user_defined_container_executor exec;

    using container_type = std::vector<
      typename std::result_of<Function(size_t)>::type
    >;

    return exec.then_execute<container_type>(f, n, fut);
  }
};


struct multi_agent_then_execute_returning_void_executor : test_executor
{
  template<class Function, class T>
  std::future<void>
    then_execute(Function f, size_t n, std::future<T>& fut)
  {
    function_called = true;

    auto val = fut.get();

    multi_agent_async_execute_returning_void_executor exec;

    // we need mutable on the lambda because we pass val to f via mutable reference
    return exec.async_execute([=](size_t idx) mutable
    {
      f(idx, val);
    },
    n);
  }

  template<class Function>
  std::future<void>
    then_execute(Function f, size_t n, std::future<void>& fut)
  {
    function_called = true;

    fut.get();

    multi_agent_async_execute_returning_void_executor exec;

    return exec.async_execute([=](size_t idx)
    {
      f(idx);
    },
    n);
  }
};


struct multi_agent_then_execute_with_shared_inits_returning_user_defined_container_executor : test_executor
{
  template<class Container, class Function, class T, class U>
  std::future<Container> then_execute(Function f, size_t n, std::future<T>& fut, U&& shared_init)
  {
    function_called = true;

    T val = fut.get(); 

    multi_agent_async_execute_with_shared_inits_returning_user_defined_container_executor exec;

    using shared_arg_type = typename std::decay<U>::type;

    return exec.async_execute<Container>([=,&val](const size_t& idx, shared_arg_type& shared_arg)
    {
      return f(idx, val, shared_arg);
    },
    n,
    std::forward<U>(shared_init));
  }

  template<class Container, class Function, class T>
  std::future<Container> then_execute(Function f, size_t n, std::future<void>& fut, T&& shared_init)
  {
    function_called = true;

    fut.get(); 

    multi_agent_async_execute_with_shared_inits_returning_user_defined_container_executor exec;

    return exec.async_execute<Container>(f, n, std::forward<T>(shared_init));
  }
};


struct multi_agent_then_execute_with_shared_inits_returning_default_container_executor : test_executor
{
  template<class Function, class T, class U>
  std::future<
    std::vector<
      typename std::result_of<Function(size_t, T&, typename std::decay<U>::type&)>::type
    >
  >
    then_execute(Function f, size_t n, std::future<T>& fut, U&& shared_init)
  {
    function_called = true;

    multi_agent_then_execute_with_shared_inits_returning_user_defined_container_executor exec;

    using container_type = std::vector<
      typename std::result_of<Function(size_t, T&, typename std::decay<U>::type&)>::type
    >;

    return exec.then_execute<container_type>(f, n, fut, std::forward<U>(shared_init));
  }

  template<class Function, class T>
  std::future<
    std::vector<
      typename std::result_of<Function(size_t, typename std::decay<T>::type&)>::type
    >
  >
    then_execute(Function f, size_t n, std::future<void>& fut, T&& shared_init)
  {
    function_called = true;

    multi_agent_then_execute_with_shared_inits_returning_user_defined_container_executor exec;

    using container_type = std::vector<
      typename std::result_of<Function(size_t, typename std::decay<T>::type&)>::type
    >;

    return exec.then_execute<container_type>(f, n, fut, std::forward<T>(shared_init));
  }
};


struct multi_agent_then_execute_with_shared_inits_returning_void_executor : test_executor
{
  template<class Function, class T, class U>
  std::future<void>
    then_execute(Function f, size_t n, std::future<T>& fut, U&& shared_init)
  {
    function_called = true;

    auto val = fut.get();

    multi_agent_async_execute_with_shared_inits_returning_void_executor exec;

    using shared_arg_type = typename std::decay<U>::type;

    // we need mutable on the lambda because we pass val to f via mutable reference
    return exec.async_execute([=](size_t idx, shared_arg_type& shared_arg) mutable
    {
      f(idx, val, shared_arg);
    },
    n,
    std::forward<U>(shared_init));
  }

  template<class Function, class T>
  std::future<void>
    then_execute(Function f, size_t n, std::future<void>& fut, T&& shared_init)
  {
    function_called = true;

    fut.get();

    multi_agent_async_execute_with_shared_inits_returning_void_executor exec;

    using shared_arg_type = typename std::decay<T>::type;

    return exec.async_execute([=](size_t idx, shared_arg_type& shared_arg)
    {
      f(idx, shared_arg);
    },
    n,
    std::forward<T>(shared_init));
  }
};


} // end test_executors 

