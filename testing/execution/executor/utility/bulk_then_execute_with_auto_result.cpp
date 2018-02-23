#include <agency/agency.hpp>
#include <agency/execution/executor/detail/utility.hpp>
#include <iostream>

#include "../test_executors.hpp"


template<class Executor>
void test_with_void_predecessor_returning_void(Executor exec)
{
  agency::executor_shape_t<Executor> shape{100};

  auto predecessor_future = agency::make_ready_future<void>(exec);
  
  size_t shared_arg = 0;
  
  size_t increment_me = 0;
  std::mutex mut;
  auto fut = agency::detail::bulk_then_execute_with_auto_result(exec, [&](size_t, size_t& shared_arg)
  {
    mut.lock();
    increment_me += 1;
    ++shared_arg;
    mut.unlock();
  },
  shape,
  predecessor_future,
  [&]
  {
    return std::ref(shared_arg);
  });
  
  fut.wait();
  
  assert(increment_me == shape);
  assert(shared_arg == shape);
}


template<class Executor>
void test_with_void_predecessor_returning_results(Executor exec)
{
  auto predecessor_future = agency::detail::make_ready_future();

  using index_type = agency::executor_index_t<Executor>;

  size_t shape = 10;
  
  auto f = agency::detail::bulk_then_execute_with_auto_result(exec,
    [](index_type idx, std::vector<int>& shared_arg)
    {
      return shared_arg[idx];
    },
    shape,
    predecessor_future,
    [=]{ return std::vector<int>(shape, 13); }  // shared_arg
  );
  
  auto result = f.get();
  
  using container_type = agency::vector<int, agency::executor_allocator_t<Executor,int>>;
  assert(container_type(shape, 13) == result);
}


template<class Executor>
void test_with_non_void_predecessor_returning_void(Executor exec)
{
  agency::executor_shape_t<Executor> shape{100};

  auto predecessor_future = agency::make_ready_future<int>(exec, 13);
  
  size_t shared_arg = 0;
  
  size_t increment_me = 0;
  std::mutex mut;
  auto fut = agency::detail::bulk_then_execute_with_auto_result(exec, [&](size_t, int& predecessor, size_t& shared_arg)
  {
    mut.lock();
    increment_me += predecessor;
    ++shared_arg;
    mut.unlock();
  },
  shape,
  predecessor_future,
  [&]
  {
    return std::ref(shared_arg);
  });
  
  fut.wait();
  
  assert(increment_me == shape * 13);
  assert(shared_arg == shape);
}


template<class Executor>
void test_with_non_void_predecessor_returning_results(Executor exec)
{
  auto predecessor_future = agency::make_ready_future<int>(exec, 7);

  using index_type = agency::executor_index_t<Executor>;

  size_t shape = 10;
  
  auto f = agency::detail::bulk_then_execute_with_auto_result(exec,
    [](index_type idx, int& predecessor, std::vector<int>& shared_arg)
    {
      return predecessor + shared_arg[idx];
    },
    shape,
    predecessor_future,
    [=]{ return std::vector<int>(shape, 13); }  // shared_arg
  );
  
  auto result = f.get();
  
  using container_type = agency::vector<int, agency::executor_allocator_t<Executor,int>>;
  assert(container_type(shape, 7 + 13) == result);
}


int main()
{
  test_with_void_predecessor_returning_void(bulk_twoway_executor());
  test_with_void_predecessor_returning_void(bulk_then_executor());
  test_with_void_predecessor_returning_void(not_a_bulk_twoway_executor());
  test_with_void_predecessor_returning_void(not_a_bulk_then_executor());
  test_with_void_predecessor_returning_void(complete_bulk_executor());

  test_with_void_predecessor_returning_results(bulk_twoway_executor());
  test_with_void_predecessor_returning_results(bulk_then_executor());
  test_with_void_predecessor_returning_results(not_a_bulk_twoway_executor());
  test_with_void_predecessor_returning_results(not_a_bulk_then_executor());
  test_with_void_predecessor_returning_results(complete_bulk_executor());

  test_with_non_void_predecessor_returning_void(bulk_twoway_executor());
  test_with_non_void_predecessor_returning_void(bulk_then_executor());
  test_with_non_void_predecessor_returning_void(not_a_bulk_twoway_executor());
  test_with_non_void_predecessor_returning_void(not_a_bulk_then_executor());
  test_with_non_void_predecessor_returning_void(complete_bulk_executor());

  test_with_non_void_predecessor_returning_results(bulk_twoway_executor());
  test_with_non_void_predecessor_returning_results(bulk_then_executor());
  test_with_non_void_predecessor_returning_results(not_a_bulk_twoway_executor());
  test_with_non_void_predecessor_returning_results(not_a_bulk_then_executor());
  test_with_non_void_predecessor_returning_results(complete_bulk_executor());

  std::cout << "OK" << std::endl;
  
  return 0;
}

