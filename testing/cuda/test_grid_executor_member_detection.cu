#include <agency/cuda/grid_executor.hpp>
#include <agency/executor_traits.hpp>
#include <iostream>

struct functor_returning_void
{
  template<class T>
  __host__ __device__
  void operator()(const T&) const {}

  __host__ __device__
  void operator()() const {}
};

int main()
{
  using executor_type = agency::cuda::grid_executor;

  using int_future_type = agency::cuda::future<int>;

  static_assert(
    agency::detail::new_executor_traits_detail::has_multi_agent_then_execute_returning_void<executor_type, functor_returning_void, int_future_type>::value,
    "grid_executor should have multi-agent then_execute() returning void"
  );

  return 0;
}

