#include <agency/agency.hpp>
#include <agency/execution/executor/detail/new_executor_traits.hpp>
#include <agency/cuda.hpp>
#include <iostream>

#include "../test_executors.hpp"


__managed__ int increment_me;


template<class Executor>
void test_with_void_predecessor_returning_void(Executor exec)
{
  using namespace agency::detail::new_executor_traits_detail;

  agency::new_executor_shape_t<Executor> shape{100};

  agency::new_executor_future_t<Executor,void> predecessor = agency::future_traits<agency::new_executor_future_t<Executor,void>>::make_ready();
  
  int shared_arg = 0;
  
  int increment_me = 0;
  std::mutex mut;
  auto fut = bulk_then_execute_with_auto_result(exec, [&](size_t idx, int& shared_arg)
  {
    mut.lock();
    increment_me += 1;
    ++shared_arg;
    mut.unlock();
  },
  shape,
  predecessor,
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
  using namespace agency::detail::new_executor_traits_detail;

  auto predecessor_future = agency::detail::make_ready_future();

  using shape_type = agency::new_executor_shape_t<Executor>;
  using index_type = agency::new_executor_index_t<Executor>;

  size_t shape = 10;
  
  auto f = bulk_then_execute_with_auto_result(exec,
    [](index_type idx, std::vector<int>& shared_arg)
    {
      return shared_arg[idx];
    },
    shape,
    predecessor_future,
    [=]{ return std::vector<int>(shape, 13); }  // shared_arg
  );
  
  auto result = f.get();
  
  using container_type = executor_container_t<Executor,int>;
  assert(container_type(shape, 13) == result);
}


template<class Executor>
void test_with_non_void_predecessor_returning_void(Executor exec)
{
  using namespace agency::detail::new_executor_traits_detail;

  agency::new_executor_shape_t<Executor> shape{100};

  agency::new_executor_future_t<Executor,int> predecessor_future = agency::future_traits<agency::new_executor_future_t<Executor,int>>::template make_ready<int>(13);
  
  int shared_arg = 0;
  
  int increment_me = 0;
  std::mutex mut;
  auto fut = bulk_then_execute_with_auto_result(exec, [&](size_t idx, int& predecessor, int& shared_arg)
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
  using namespace agency::detail::new_executor_traits_detail;

  agency::new_executor_future_t<Executor,int> predecessor_future = agency::future_traits<agency::new_executor_future_t<Executor,int>>::template make_ready<int>(7);

  using shape_type = agency::new_executor_shape_t<Executor>;
  using index_type = agency::new_executor_index_t<Executor>;

  size_t shape = 10;
  
  auto f = bulk_then_execute_with_auto_result(exec,
    [](index_type idx, int& predecessor, std::vector<int>& shared_arg)
    {
      return predecessor + shared_arg[idx];
    },
    shape,
    predecessor_future,
    [=]{ return std::vector<int>(shape, 13); }  // shared_arg
  );
  
  auto result = f.get();
  
  using container_type = executor_container_t<Executor,int>;
  assert(container_type(shape, 7 + 13) == result);
}


template<class Executor>
void test_with_void_predecessor_returning_void2(Executor exec)
{
  using namespace agency::detail::new_executor_traits_detail;

  agency::new_executor_shape_t<Executor> shape{10,10};

  agency::new_executor_future_t<Executor,void> predecessor = agency::future_traits<agency::new_executor_future_t<Executor,void>>::make_ready();

  increment_me = 0;

  using index_type = agency::new_executor_index_t<Executor>;
  
  auto fut = bulk_then_execute_with_auto_result(exec, [] __device__ (index_type idx, int& outer_shared_arg, int& inner_shared_arg)
  {
    atomicAdd(&increment_me, outer_shared_arg + inner_shared_arg);
  },
  shape,
  predecessor,
  [] __host__ __device__ { return 7; },
  [] __host__ __device__ { return 13; }
  );
  
  fut.wait();
  
  assert(increment_me == shape[0] * shape[1] * (7 + 13));
}


template<class Executor>
void test_with_void_predecessor_returning_results2(Executor exec)
{
  using namespace agency::detail::new_executor_traits_detail;

  agency::new_executor_future_t<Executor,void> predecessor_future = agency::future_traits<agency::new_executor_future_t<Executor,void>>::make_ready();

  using shape_type = agency::new_executor_shape_t<Executor>;
  using index_type = agency::new_executor_index_t<Executor>;

  shape_type shape{10,10};
  
  auto f = bulk_then_execute_with_auto_result(exec,
    [] __host__ __device__ (index_type idx, int& outer_shared_arg, int& inner_shared_arg)
    {
      return outer_shared_arg + inner_shared_arg;
    },
    shape,
    predecessor_future,
    [] __host__ __device__ { return 7; }, // outer_shared_arg
    [] __host__ __device__ { return 13; }   // inner_shared_arg
  );
  
  auto result = f.get();
  
  using container_type = executor_container_t<Executor,int>;
  assert(container_type(shape, 7 + 13) == result);
}


template<class Executor>
void test_with_non_void_predecessor_returning_void2(Executor exec)
{
  using namespace agency::detail::new_executor_traits_detail;

  agency::new_executor_shape_t<Executor> shape{10,10};

  agency::new_executor_future_t<Executor,int> predecessor_future = agency::future_traits<agency::new_executor_future_t<Executor,int>>::template make_ready(42);

  increment_me = 0;

  using index_type = agency::new_executor_index_t<Executor>;
  
  auto fut = bulk_then_execute_with_auto_result(exec, [] __device__ (index_type idx, int& predecessor, int& outer_shared_arg, int& inner_shared_arg)
  {
    atomicAdd(&increment_me, predecessor + outer_shared_arg + inner_shared_arg);
  },
  shape,
  predecessor_future,
  [] __host__ __device__ { return 7; },
  [] __host__ __device__ { return 13; }
  );
  
  fut.wait();
  
  assert(increment_me == shape[0] * shape[1] * (42 + 7 + 13));
}


template<class Executor>
void test_with_non_void_predecessor_returning_results2(Executor exec)
{
  using namespace agency::detail::new_executor_traits_detail;

  agency::new_executor_shape_t<Executor> shape{10,10};

  agency::new_executor_future_t<Executor,int> predecessor_future = agency::future_traits<agency::new_executor_future_t<Executor,int>>::template make_ready(42);

  using index_type = agency::new_executor_index_t<Executor>;
  
  auto fut = bulk_then_execute_with_auto_result(exec, [] __host__ __device__ (index_type idx, int& predecessor, int& outer_shared_arg, int& inner_shared_arg)
  {
    return predecessor + outer_shared_arg + inner_shared_arg;
  },
  shape,
  predecessor_future,
  [] __host__ __device__ { return 7; },
  [] __host__ __device__ { return 13; }
  );
  
  auto result = fut.get();
  
  using container_type = executor_container_t<Executor,int>;
  assert(container_type(shape, 42 + 7 + 13) == result);
}


int main()
{
  test_with_void_predecessor_returning_void(bulk_synchronous_executor());
  test_with_void_predecessor_returning_void(bulk_asynchronous_executor());
  test_with_void_predecessor_returning_void(bulk_continuation_executor());
  test_with_void_predecessor_returning_void(not_a_bulk_synchronous_executor());
  test_with_void_predecessor_returning_void(not_a_bulk_asynchronous_executor());
  test_with_void_predecessor_returning_void(not_a_bulk_continuation_executor());
  test_with_void_predecessor_returning_void(complete_bulk_executor());

  test_with_void_predecessor_returning_results(bulk_synchronous_executor());
  test_with_void_predecessor_returning_results(bulk_asynchronous_executor());
  test_with_void_predecessor_returning_results(bulk_continuation_executor());
  test_with_void_predecessor_returning_results(not_a_bulk_synchronous_executor());
  test_with_void_predecessor_returning_results(not_a_bulk_asynchronous_executor());
  test_with_void_predecessor_returning_results(not_a_bulk_continuation_executor());
  test_with_void_predecessor_returning_results(complete_bulk_executor());

  test_with_non_void_predecessor_returning_void(bulk_synchronous_executor());
  test_with_non_void_predecessor_returning_void(bulk_asynchronous_executor());
  test_with_non_void_predecessor_returning_void(bulk_continuation_executor());
  test_with_non_void_predecessor_returning_void(not_a_bulk_synchronous_executor());
  test_with_non_void_predecessor_returning_void(not_a_bulk_asynchronous_executor());
  test_with_non_void_predecessor_returning_void(not_a_bulk_continuation_executor());
  test_with_non_void_predecessor_returning_void(complete_bulk_executor());

  test_with_non_void_predecessor_returning_results(bulk_synchronous_executor());
  test_with_non_void_predecessor_returning_results(bulk_asynchronous_executor());
  test_with_non_void_predecessor_returning_results(bulk_continuation_executor());
  test_with_non_void_predecessor_returning_results(not_a_bulk_synchronous_executor());
  test_with_non_void_predecessor_returning_results(not_a_bulk_asynchronous_executor());
  test_with_non_void_predecessor_returning_results(not_a_bulk_continuation_executor());
  test_with_non_void_predecessor_returning_results(complete_bulk_executor());


  test_with_void_predecessor_returning_void2(agency::cuda::grid_executor());
  test_with_void_predecessor_returning_results2(agency::cuda::grid_executor());
  test_with_non_void_predecessor_returning_void2(agency::cuda::grid_executor());
  test_with_non_void_predecessor_returning_results2(agency::cuda::grid_executor());

  std::cout << "OK" << std::endl;
  
  return 0;
}

