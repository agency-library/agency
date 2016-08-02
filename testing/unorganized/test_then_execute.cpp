#include <agency/agency.hpp>
#include <atomic>
#include <algorithm>
#include <cassert>

template<class Executor, class Shape>
void test_nested(Shape shape)
{
  using executor_type = Executor;
  using traits = agency::executor_traits<executor_type>;
  using index_type = typename traits::index_type;
  using shape_type = typename traits::shape_type;
  
  executor_type exec;
  
  {
    // no return, no past parameter, no shared parameter
    auto f1 = traits::template make_ready_future<void>(exec);
    
    std::atomic<int> counter(0);
    auto f2 = traits::then_execute(exec, [&counter](index_type idx)
    {
      counter++;
    },
    shape,
    f1
    );
    
    f2.wait();
    assert(counter == agency::detail::shape_cast<size_t>(shape));
  }

  {
    // no return, past parameter, no shared parameter
    auto f1 = traits::template make_ready_future<int>(exec, 13);

    std::atomic<int> counter(0);
    auto f2 = traits::then_execute(exec, [&counter](index_type idx, int& past_parameter)
    {
      counter++;
    },
    shape,
    f1
    );

    f2.wait();
    assert(counter == agency::detail::shape_cast<size_t>(shape));
  }
  
  {
    // no return, no past parameter, shared parameter
    auto f1 = traits::template make_ready_future<void>(exec);

    std::atomic<int> result(0);
    auto f2 = traits::then_execute(exec, [&result](index_type idx, int& shared_parameter1, int& shared_parameter2)
    {
      result += shared_parameter1 + shared_parameter2;
    },
    shape,
    f1,
    []{ return 7; },
    []{ return 7; }
    );

    f2.wait();

    assert(result == (7 + 7) * agency::detail::shape_cast<size_t>(shape));
  }

  {
    // no return, past parameter, shared parameter, result
    auto f1 = traits::template make_ready_future<int>(exec, 13);

    std::atomic<int> result(0);
    auto f2 = traits::then_execute(exec, [&result](index_type idx, int& past_parameter, int& shared_parameter1, int& shared_parameter2)
    {
      result += past_parameter + shared_parameter1 + shared_parameter2;
    },
    shape,
    f1,
    []{ return 7; },
    []{ return 7; }
    );

    f2.wait();

    assert(result == (13 + 7 + 7) * agency::detail::shape_cast<size_t>(shape));
  }
  
  // XXX not implemented yet
  //{
  //  // return, no past parameter, shared parameter
  //  auto f1 = traits::template make_ready_future<void>(exec);

  //  auto f2 = traits::then_execute(exec, [](index_type idx, int& shared_parameter1, int& shared_parameter2)
  //  {
  //    return shared_parameter1 + shared_parameter2;
  //  },
  //  [](shape_type shape)
  //  {
  //    return typename traits::template container<int>(shape);
  //  },
  //  shape,
  //  f1,
  //  []{ return 7; },
  //  []{ return 7; }
  //  );

  //  auto result = f2.get();
  //  assert(std::all_of(result.begin(), result.end(), [](int x) { return x == 7; }));
  //  assert(result.size() == agency::detail::shape_cast<size_t>(shape));
  //}

  // XXX not implemented yet
  //{
  //  // return, past parameter, shared parameter
  //  auto f1 = traits::template make_ready_future<int>(exec, 13);

  //  auto f2 = traits::then_execute(exec, [](index_type idx, int& past_parameter, int& shared_parameter)
  //  {
  //    return past_parameter + shared_parameter;
  //  },
  //  [](shape_type shape)
  //  {
  //    return typename traits::template container<int>(shape);
  //  },
  //  shape,
  //  f1,
  //  []{ return 7; }
  //  );

  //  auto result = f2.get();
  //  assert(std::all_of(result.begin(), result.end(), [](int x) { return x == 20; }));
  //  assert(result.size() == agency::detail::shape_cast<size_t>(shape));
  //}
}

template<class Executor, class Shape>
void test(Shape shape)
{
  using executor_type = Executor;
  using traits = agency::executor_traits<executor_type>;
  using index_type = typename traits::index_type;
  using shape_type = typename traits::shape_type;
  
  executor_type exec;
  
  {
    // no return, no past parameter, no shared parameter
    auto f1 = traits::template make_ready_future<void>(exec);
    
    std::atomic<int> counter(0);
    auto f2 = traits::then_execute(exec, [&counter](index_type idx)
    {
      counter++;
    },
    shape,
    f1
    );
    
    f2.wait();
    assert(counter == agency::detail::shape_cast<size_t>(shape));
  }

  {
    // no return, past parameter, no shared parameter
    auto f1 = traits::template make_ready_future<int>(exec, 13);

    std::atomic<int> counter(0);
    auto f2 = traits::then_execute(exec, [&counter](index_type idx, int& past_parameter)
    {
      counter++;
    },
    shape,
    f1
    );

    f2.wait();
    assert(counter == agency::detail::shape_cast<size_t>(shape));
  }
  
  {
    // no return, no past parameter, shared parameter
    auto f1 = traits::template make_ready_future<void>(exec);

    std::atomic<int> result(0);
    auto f2 = traits::then_execute(exec, [&result](index_type idx, int& shared_parameter)
    {
      result += shared_parameter;
    },
    shape,
    f1,
    []{ return 7; }
    );

    f2.wait();

    assert(result == 7 * agency::detail::shape_cast<size_t>(shape));
  }

  {
    // no return, past parameter, shared parameter, result
    auto f1 = traits::template make_ready_future<int>(exec, 13);

    std::atomic<int> result(0);
    auto f2 = traits::then_execute(exec, [&result](index_type idx, int& past_parameter, int& shared_parameter)
    {
      result += past_parameter + shared_parameter;
    },
    shape,
    f1,
    []{ return 7; }
    );

    f2.wait();

    assert(result == 20 * agency::detail::shape_cast<size_t>(shape));
  }
  
  {
    // return, no past parameter, shared parameter
    auto f1 = traits::template make_ready_future<void>(exec);

    auto f2 = traits::then_execute(exec, [](index_type idx, int& shared_parameter)
    {
      return shared_parameter;
    },
    [](shape_type shape)
    {
      return typename traits::template container<int>(shape);
    },
    shape,
    f1,
    []{ return 7; }
    );

    auto result = f2.get();
    assert(std::all_of(result.begin(), result.end(), [](int x) { return x == 7; }));
    assert(result.size() == agency::detail::shape_cast<size_t>(shape));
  }

  {
    // return, past parameter, shared parameter
    auto f1 = traits::template make_ready_future<int>(exec, 13);

    auto f2 = traits::then_execute(exec, [](index_type idx, int& past_parameter, int& shared_parameter)
    {
      return past_parameter + shared_parameter;
    },
    [](shape_type shape)
    {
      return typename traits::template container<int>(shape);
    },
    shape,
    f1,
    []{ return 7; }
    );

    auto result = f2.get();
    assert(std::all_of(result.begin(), result.end(), [](int x) { return x == 20; }));
    assert(result.size() == agency::detail::shape_cast<size_t>(shape));
  }
}

int main()
{
  test<agency::sequential_executor>(10);
  test<agency::parallel_executor>(10);
  test<agency::concurrent_executor>(10);

  using executor_type = agency::scoped_executor<agency::concurrent_executor,agency::sequential_executor>;
  test_nested<executor_type>(executor_type::shape_type(4,4));

  std::cout << "OK" << std::endl;

  return 0;
}

