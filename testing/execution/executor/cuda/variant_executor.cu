#include <agency/execution/executor/variant_executor.hpp>
#include <agency/execution/executor/sequenced_executor.hpp>
#include <agency/execution/executor/parallel_executor.hpp>
#include <agency/execution/executor/concurrent_executor.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/experimental/ndarray.hpp>
#include <agency/coordinate/detail/colexicographic_rank.hpp>

#include <agency/cuda/execution/executor/parallel_executor.hpp>
#include <agency/cuda/execution/executor/concurrent_executor.hpp>

#include <iostream>
#include <atomic>
#include <functional>
#include <numeric>


template<class... Args>
void swallow(Args&&...) {}


__managed__ unsigned int counter;

__host__ __device__ 
void atomic_increment_counter(unsigned int& value, unsigned int increment = 1)
{
#ifndef __CUDA_ARCH__
  value += increment;
#else
  atomicAdd(&value, increment);
#endif
}


struct return_one
{
  __host__ __device__
  int operator()() const { return 1; }
};


template<class Container, class Allocator>
struct return_container
{
  typename Container::shape_type shape;
  Allocator alloc;

  __agency_exec_check_disable__
  __host__ __device__
  return_container(typename Container::shape_type s, Allocator a)
    : shape(s), alloc(a)
  {}

  __agency_exec_check_disable__
  return_container(const return_container&) = default;

  __agency_exec_check_disable__
  ~return_container() = default;

  __host__ __device__
  Container operator()() const
  {
    return Container(shape, alloc);
  }
};


template<class VariantExecutor, class Executor>
int test_alternative(Executor alternative)
{
  using namespace agency;

  VariantExecutor exec = alternative;
  using shape_type = executor_shape_t<VariantExecutor>;
  using index_type = executor_index_t<VariantExecutor>;

  {
    // test .type()
    assert(exec.type() == typeid(alternative));
  }

  {
    // test twoway_execute()
    counter = 0;
    auto void_future = exec.twoway_execute([] __host__ __device__
    {
      counter = 1;
    });

    void_future.wait();

    assert(counter);

    counter = 13;

    auto int_future = exec.twoway_execute([] __host__ __device__
    {
      return counter;
    });

    assert(counter == int_future.get());
  }

  {
    // test bulk_twoway_execute()
    using int_container = agency::experimental::basic_ndarray<int, shape_type, agency::executor_allocator_t<VariantExecutor, int>>;

    size_t num_agents = 10;

    shape_type shape = detail::shape_cast<shape_type>(num_agents);

    // create an allocator corresponding to the selected alternative
    executor_allocator_t<Executor,int> alternative_alloc;

    counter = 0;

    auto future_results = exec.bulk_twoway_execute([] __host__ __device__ (index_type idx, int_container& results, int& inc)
    {
      results[idx] = detail::colexicographic_rank(idx, results.shape());

      atomic_increment_counter(counter, inc);
    },
    shape,

    // XXX this lambda causes __host__ __device__ warnings so use a functor
    //[=] __host__ __device__ { return int_container(shape, alternative_alloc); }, // result factory
    return_container<int_container, executor_allocator_t<Executor,int>>(shape, alternative_alloc),

    []  __host__ __device__ { return 1; }                                          // shared factory
    );

    int_container results = future_results.get();

    int_container reference(shape);
    std::iota(reference.begin(), reference.end(), 0);

    assert(num_agents == counter);
    assert(reference == results);
  }

  {
    // test bulk_then_execute()
    using int_container = agency::experimental::basic_ndarray<int, shape_type, agency::executor_allocator_t<VariantExecutor, int>>;

    int predecessor = 7;
    auto predecessor_future = exec.template make_ready_future<int>(predecessor);

    size_t num_agents = 10;

    shape_type shape = detail::shape_cast<shape_type>(num_agents);

    // create an allocator corresponding to the selected alternative
    executor_allocator_t<Executor,int> alternative_alloc;

    counter = 0;

    auto future_results = exec.bulk_then_execute([] __host__ __device__ (index_type idx, int& predecessor, int_container& results, int& factor)
    {
      results[idx] = detail::colexicographic_rank(idx, results.shape());

      atomic_increment_counter(counter, factor * predecessor);
    },
    shape,
    predecessor_future,                                                            // future

    // XXX nvcc miscompiles this lambda for some reason
    //[=] __host__ __device__ { return int_container(shape, alternative_alloc); }, // result factory
    return_container<int_container, executor_allocator_t<Executor,int>>(shape, alternative_alloc),

    // XXX nvcc miscompiles this lambda for some reason
    //[] __host__ __device__ { return 1; }                                         // shared factory
    return_one()
    );

    auto results = future_results.get();

    int_container reference(shape);
    std::iota(reference.begin(), reference.end(), 0);

    assert(predecessor * num_agents == counter);
    assert(reference == results);
  }

  // XXX this test is disabled because then_execute(sequenced_executor, ...) isn't implemented
  //{
  //  // test future_cast()
  //  int value = 7;
  //  auto int_future = exec.template make_ready_future<int>(value);

  //  // cast to float
  //  auto float_future = exec.template future_cast<float>(int_future);

  //  float result = float_future.get();

  //  assert(result == float(value));
  //}

  {
    // test max_shape_dimensions()
    auto reference = agency::max_shape_dimensions(alternative);

    auto result = exec.max_shape_dimensions();

    assert(reference == detail::shape_cast<decltype(reference)>(result));
  }

  // XXX these tests are disabled because then_execute(sequenced_executor, ...) isn't implemented
  //{
  //  // test then_execute()
  //  
  //  {
  //    // void predecessor case
  //    auto predecessor_future1 = exec.template make_ready_future<void>();

  //    bool executed = false;
  //    auto void_future = exec.then_execute([&]
  //    {
  //      executed = true;
  //    },
  //    predecessor_future1
  //    );

  //    void_future.wait();

  //    assert(executed);

  //    auto predecessor_future2 = exec.template make_ready_future<void>();

  //    int reference = 13;

  //    auto int_future = exec.then_execute([=]
  //    {
  //      return reference;
  //    },
  //    predecessor_future2
  //    );

  //    assert(reference == int_future.get());
  //  }

  //  {
  //    // int predecessor case
  //    int reference = 7;

  //    auto predecessor_future1 = exec.template make_ready_future<int>(reference);

  //    int result1 = 0;
  //    auto void_future = exec.then_execute([&](int& predecessor)
  //    {
  //      result1 = predecessor;
  //    },
  //    predecessor_future1
  //    );

  //    void_future.wait();

  //    assert(reference == result1);

  //    auto predecessor_future2 = exec.template make_ready_future<int>(reference);

  //    auto int_future = exec.then_execute([=](int& predecessor)
  //    {
  //      return predecessor;
  //    },
  //    predecessor_future2
  //    );

  //    assert(reference == int_future.get());
  //  }
  //}

  {
    // test unit_shape()
    auto reference = agency::unit_shape(alternative);

    auto result = exec.unit_shape();

    assert(reference == detail::shape_cast<decltype(reference)>(result));
  }

  return 0;
}


template<class... Executors>
void test(Executors... execs)
{
  using namespace agency;

  using executor_type = variant_executor<Executors...>;

  static_assert(is_executor<executor_type>::value, "variant_executor is not an executor");
  static_assert(detail::is_single_twoway_executor<executor_type>::value, "variant_executor is not a single twoway executor");
  static_assert(detail::is_single_then_executor<executor_type>::value, "variant_executor is not a single then executor");
  static_assert(detail::is_bulk_twoway_executor<executor_type>::value, "variant_executor is not a bulk twoway executor");
  static_assert(detail::is_bulk_then_executor<executor_type>::value, "variant_executor is not a bulk then executor");

  std::tuple<Executors...> executor_tuple(execs...);

  // test with each type of alternative executor
  swallow(test_alternative<executor_type>(execs)...);
}


int main()
{
  {
    // test with a single alternative
    test(agency::sequenced_executor());
  }

  {
    // test with two alternatives
    test(agency::sequenced_executor(), agency::cuda::parallel_executor());
  }

  // XXX this takes too long to compile
  //{
  //  // test with three alternatives
  //  test(agency::sequenced_executor(), agency::cuda::parallel_executor(), agency::cuda::concurrent_executor());
  //}

  std::cout << "OK" << std::endl;
}

