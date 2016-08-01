#include <agency/cuda.hpp>
#include <algorithm>
#include <iostream>


__managed__ int increment_me;

struct increment_and_return_void
{
  template<class Index>
  __device__
  void operator()(const Index&, int& past, int& outer_shared_arg, int& inner_shared_arg)
  {
    atomicAdd(&increment_me, past + outer_shared_arg + inner_shared_arg);
  }
};


struct sum_13_7_and_42
{
  template<class Index, class T1, class T2, class T3>
  __device__
  int operator()(const Index&, T1& past, T2& outer_shared_arg, T3& inner_shared_arg)
  {
    assert(past == 13);
    assert(outer_shared_arg == 7);
    assert(inner_shared_arg == 42);
    return past + outer_shared_arg + inner_shared_arg;
  }
};


template<class T>
struct factory
{
  __host__ __device__
  T operator()() const
  {
    return value;
  }

  T value;
};

template<class T>
__host__ __device__
factory<T> make_factory(const T& value)
{
  return factory<T>{value};
}


template<class Container>
struct container_factory
{
  template<class Shape>
  __host__ __device__
  Container operator()(const Shape& shape)
  {
    return Container(shape);
  }
};


template<class Executor>
void test()
{
  using executor_type = Executor;

  using traits = agency::executor_traits<executor_type>;

  {
    // then_execute returning user-specified container
    
    executor_type exec;

    typename executor_type::shape_type shape{10,10};

    auto past = traits::template make_ready_future<int>(exec, 13);

    using container_type = typename executor_type::template container<int>;

    using index_type = typename executor_type::index_type;

    auto fut = traits::then_execute(exec, sum_13_7_and_42(), container_factory<container_type>{}, shape, past, make_factory(7), make_factory(42.0f));

    auto got = fut.get();

    assert(got == std::vector<int>(got.size(), 13 + 7 + 42));
  }

  {
    // then_execute returning default container
    
    executor_type exec;

    typename executor_type::shape_type shape{10,10};

    auto past = traits::template make_ready_future<int>(exec, 13);

    using index_type = typename traits::index_type;

    auto fut = traits::then_execute(exec, sum_13_7_and_42(), shape, past, make_factory(7), make_factory(42.0f));

    auto result = fut.get();

    std::vector<int> ref(result.size(), 13 + 7 + 42);
    assert(std::equal(ref.begin(), ref.end(), result.begin()));
  }

  {
    // then_execute returning void
    
    executor_type exec;

    typename executor_type::shape_type shape{10,10};

    auto past = traits::template make_ready_future<int>(exec, 13);

    increment_me = 0;

    using index_type = typename traits::index_type;

    auto fut = traits::then_execute(exec, increment_and_return_void(), shape, past, make_factory(7), make_factory(42));

    fut.wait();

    assert(increment_me == shape[0] * shape[1] * (13 + 7 + 42));
  }
}

int main()
{
  // a completely empty executor
  test<agency::cuda::grid_executor>();

  std::cout << "OK" << std::endl;

  return 0;
}

