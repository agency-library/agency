#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <memory>


int main()
{
  auto factory = [] __host__ __device__ { return 7; };

  agency::cuda::grid_executor exec;
  using traits = agency::executor_traits<agency::cuda::grid_executor>;
  auto shape = agency::cuda::grid_executor::shape_type{100,256};

  {
    // int, float -> (int, float)
    auto f1 = traits::make_ready_future<int>(exec,7);
    auto f2 = traits::make_ready_future<float>(exec,13);

    auto f3 = exec.when_all_execute_and_select<0,1>([] __host__ __device__ (agency::cuda::grid_executor::index_type idx, int& past_arg1, float& past_arg2, int& outer_arg, int& inner_arg)
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
    auto f1 = traits::make_ready_future<int>(exec,7);
    auto f2 = traits::make_ready_future<float>(exec,13);

    auto f3 = exec.when_all_execute_and_select<1,0>([] __host__ __device__ (agency::cuda::grid_executor::index_type idx, int& past_arg1, float& past_arg2, int& outer_arg, int& inner_arg)
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
    auto f1 = traits::make_ready_future<int>(exec,7);
    auto f2 = traits::make_ready_future<float>(exec,13);

    auto f3 = exec.when_all_execute_and_select<0>([] __host__ __device__ (agency::cuda::grid_executor::index_type idx, int& past_arg1, float& past_arg2, int& outer_arg, int& inner_arg)
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
    auto f1 = traits::make_ready_future<int>(exec,7);
    auto f2 = traits::make_ready_future<float>(exec,13);

    auto f3 = exec.when_all_execute_and_select<1>([] __host__ __device__ (agency::cuda::grid_executor::index_type idx, int& past_arg1, float& past_arg2, int& outer_arg, int& inner_arg)
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
    auto f1 = traits::make_ready_future<int>(exec,7);
    auto f2 = traits::make_ready_future<void>(exec);

    auto f3 = exec.when_all_execute_and_select<0,1>([] __host__ __device__ (agency::cuda::grid_executor::index_type idx, int& past_arg, int& outer_arg, int& inner_arg)
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
    auto f1 = traits::make_ready_future<void>(exec);
    auto f2 = traits::make_ready_future<int>(exec,7);

    auto f3 = exec.when_all_execute_and_select<0,1>([] __host__ __device__ (agency::cuda::grid_executor::index_type idx, int& past_arg, int& outer_arg, int& inner_arg)
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
    auto f1 = traits::make_ready_future<void>(exec);

    auto f3 = exec.when_all_execute_and_select<0>([] __host__ __device__ (agency::cuda::grid_executor::index_type idx, int& outer_arg, int& inner_arg)
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
    auto f1 = traits::make_ready_future<void>(exec);
    auto f2 = traits::make_ready_future<void>(exec);

    auto f3 = exec.when_all_execute_and_select<0,1>([] __host__ __device__ (agency::cuda::grid_executor::index_type idx, int& outer_arg, int& inner_arg)
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

    auto f3 = exec.when_all_execute_and_select([] __host__ __device__ (agency::cuda::grid_executor::index_type idx, int& outer_arg, int& inner_arg)
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

