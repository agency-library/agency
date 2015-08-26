#include <agency/cuda/grid_executor.hpp>
#include <agency/executor_traits.hpp>
#include <iostream>

struct functor_returning_int
{
  template<class... Args>
  __host__ __device__
  int operator()(Args&&...) const
  {
    return 0;
  }
};

struct functor_returning_void
{
  template<class T>
  __host__ __device__
  void operator()(const T&) const {}

  __host__ __device__
  void operator()() const {}
};


template<class T>
struct container
{
  template<class Shape>
  __host__ __device__
  container(const Shape&);

  template<class Index>
  __host__ __device__
  T& operator[](const Index&);
};


struct int_factory
{
  __host__ __device__
  int operator()() const { return 0; }
};


int main()
{
  using executor_type = agency::cuda::grid_executor;

  using int_future_type = agency::cuda::future<int>;

  static_assert(
    agency::detail::new_executor_traits_detail::has_multi_agent_then_execute_returning_void<executor_type, functor_returning_void, int_future_type>::value,
    "grid_executor should have multi-agent then_execute() returning void"
  );

  static_assert(
    agency::detail::new_executor_traits_detail::has_multi_agent_then_execute_with_shared_inits_returning_user_specified_container<container<int>, executor_type, functor_returning_int, int_future_type, int_factory, int_factory>::value,
    "grid_executor should have multi-agent then_execute() with shared inits returning user-specified container"
  );

  std::cout << "OK" << std::endl;

  return 0;
}

