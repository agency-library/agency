#include <agency/agency.hpp>
#include <agency/cuda.hpp>
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


template<class Container>
struct container_factory
{
  template<class Shape>
  __host__ __device__
  Container operator()(const Shape& shape) const
  {
    return Container(shape);
  }
};


struct int_factory
{
  __host__ __device__
  int operator()() const { return 0; }
};


int main()
{
  using executor_type = agency::cuda::grid_executor;
  using int_future_type = agency::cuda::async_future<int>;

  static_assert(
    agency::detail::executor_traits_detail::has_multi_agent_then_execute_with_shared_inits_returning_user_specified_container<executor_type, functor_returning_int, container_factory<container<int>>, int_future_type, int_factory, int_factory>::value,
    "grid_executor should have multi-agent then_execute() with shared inits returning user-specified container"
  );

  std::cout << "OK" << std::endl;

  return 0;
}

