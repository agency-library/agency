#include <agency/cuda/grid_executor.hpp>
#include <agency/cuda/detail/when_all_execute_and_select.hpp>
#include <memory>


int main()
{
  auto factory = [] __device__ { return 7; };

  agency::cuda::grid_executor exec;
  auto shape = agency::cuda::grid_executor::shape_type{100,256};

  {
    // int, float -> (int, float)
    agency::cuda::future<int>   f1 = agency::cuda::make_ready_future<int>(7);
    agency::cuda::future<float> f2 = agency::cuda::make_ready_future<float>(13);

    auto f3 = exec.when_all_execute_and_select<0,1>([] __device__ (agency::cuda::grid_executor::index_type idx, int& past_arg1, float& past_arg2, int& outer_arg, int& inner_arg)
    {
      if(idx[0] == 0 && idx[1] == 0)
      {
        ++past_arg1;
        ++past_arg2;
      }
    },
    shape,
    agency::detail::make_tuple(std::move(f1),std::move(f2)),
    factory,
    factory);

    auto result = f3.get();

    assert(result == agency::detail::make_tuple(8,14));
  }

  {
    // int, float -> (float, int)
    agency::cuda::future<int>   f1 = agency::cuda::make_ready_future<int>(7);
    agency::cuda::future<float> f2 = agency::cuda::make_ready_future<float>(13);

    auto f3 = exec.when_all_execute_and_select<1,0>([] __device__ (agency::cuda::grid_executor::index_type idx, int& past_arg1, float& past_arg2, int& outer_arg, int& inner_arg)
    {
      if(idx[0] == 0 && idx[1] == 0)
      {
        ++past_arg1;
        ++past_arg2;
      }
    },
    shape,
    agency::detail::make_tuple(std::move(f1),std::move(f2)),
    factory,
    factory);

    auto result = f3.get();

    assert(result == agency::detail::make_tuple(14,8));
  }

  {
    // int, float -> int
    agency::cuda::future<int>   f1 = agency::cuda::make_ready_future<int>(7);
    agency::cuda::future<float> f2 = agency::cuda::make_ready_future<float>(13);

    auto f3 = exec.when_all_execute_and_select<0>([] __device__ (agency::cuda::grid_executor::index_type idx, int& past_arg1, float& past_arg2, int& outer_arg, int& inner_arg)
    {
      if(idx[0] == 0 && idx[1] == 0)
      {
        ++past_arg1;
        ++past_arg2;
      }
    },
    shape,
    agency::detail::make_tuple(std::move(f1),std::move(f2)),
    factory,
    factory);

    auto result = f3.get();

    assert(result == 8);
  }

  {
    // int, float -> float
    agency::cuda::future<int>   f1 = agency::cuda::make_ready_future<int>(7);
    agency::cuda::future<float> f2 = agency::cuda::make_ready_future<float>(13);

    auto f3 = exec.when_all_execute_and_select<1>([] __device__ (agency::cuda::grid_executor::index_type idx, int& past_arg1, float& past_arg2, int& outer_arg, int& inner_arg)
    {
      if(idx[0] == 0 && idx[1] == 0)
      {
        ++past_arg1;
        ++past_arg2;
      }
    },
    shape,
    agency::detail::make_tuple(std::move(f1),std::move(f2)),
    factory,
    factory);

    auto result = f3.get();

    assert(result == 14);
  }

  {
    // int, void -> int
    agency::cuda::future<int>  f1 = agency::cuda::make_ready_future<int>(7);
    agency::cuda::future<void> f2 = agency::cuda::make_ready_future();

    auto f3 = exec.when_all_execute_and_select<0,1>([] __device__ (agency::cuda::grid_executor::index_type idx, int& past_arg, int& outer_arg, int& inner_arg)
    {
      if(idx[0] == 0 && idx[1] == 0)
      {
        ++past_arg;
      }
    },
    shape,
    agency::detail::make_tuple(std::move(f1),std::move(f2)),
    factory,
    factory);

    auto result = f3.get();

    assert(result == 8);
  }

  {
    // void, int -> int
    agency::cuda::future<void> f1 = agency::cuda::make_ready_future();
    agency::cuda::future<int>  f2 = agency::cuda::make_ready_future<int>(7);

    auto f3 = exec.when_all_execute_and_select<0,1>([] __device__ (agency::cuda::grid_executor::index_type idx, int& past_arg, int& outer_arg, int& inner_arg)
    {
      if(idx[0] == 0 && idx[1] == 0)
      {
        ++past_arg;
      }
    },
    shape,
    agency::detail::make_tuple(std::move(f1),std::move(f2)),
    factory,
    factory);

    auto result = f3.get();

    assert(result == 8);
  }

  {
    // void -> void
    agency::cuda::future<void> f1 = agency::cuda::make_ready_future();

    auto f3 = exec.when_all_execute_and_select<0>([] __device__ (agency::cuda::grid_executor::index_type idx, int& outer_arg, int& inner_arg)
    {
    },
    shape,
    agency::detail::make_tuple(std::move(f1)),
    factory,
    factory);

    f3.get();
  }

  {
    // void, void -> void
    agency::cuda::future<void> f1 = agency::cuda::make_ready_future();
    agency::cuda::future<void> f2 = agency::cuda::make_ready_future();

    auto f3 = exec.when_all_execute_and_select<0,1>([] __device__ (agency::cuda::grid_executor::index_type idx, int& outer_arg, int& inner_arg)
    {
    },
    shape,
    agency::detail::make_tuple(std::move(f1), std::move(f2)),
    factory,
    factory);

    f3.get();
  }

  {
    // -> void

    auto f3 = exec.when_all_execute_and_select([] __device__ (agency::cuda::grid_executor::index_type idx, int& outer_arg, int& inner_arg)
    {
    },
    shape,
    agency::detail::make_tuple(),
    factory,
    factory);

    f3.get();
  }

  std::cout << "OK" << std::endl;

  return 0;
}

