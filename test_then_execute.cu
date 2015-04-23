#include <agency/executor_traits.hpp>
#include <agency/sequential_executor.hpp>
#include <agency/parallel_executor.hpp>
#include <agency/concurrent_executor.hpp>
#include <agency/cuda/grid_executor.hpp>
#include <mutex>
#include <thread>
#include <cassert>

template<class Executor, class Shape>
void test_host(Shape shape)
{
  using executor_type = Executor;
  using traits = agency::executor_traits<executor_type>;
  using index_type = typename traits::index_type;
  
  executor_type exec;
  
  {
    auto f1 = traits::make_ready_future(exec);
    
    std::mutex mut;
    auto f2 = traits::then_execute(exec, f1, [&mut](index_type idx)
    {
      mut.lock();
      std::cout << "agent " << idx << " in thread " << std::this_thread::get_id() << " has no past parameter" << std::endl;
      mut.unlock();
    },
    shape
    );
    
    f2.wait();
  }

  {
    auto f1 = traits::make_ready_future(exec, 13);

    std::mutex mut;
    auto f2 = traits::then_execute(exec, f1, [&mut](index_type idx, int& past_parameter)
    {
      mut.lock();
      std::cout << "agent " << idx << " in thread " << std::this_thread::get_id() << " sees past_parameter " << past_parameter << std::endl;
      mut.unlock();
    },
    shape
    );

    f2.wait();
  }
}

struct device_functor
{
  template<class Index>
  __device__
  void operator()(Index idx)
  {
    printf("CUDA agent {%d, %d} has no past parameter\n", idx[0], idx[1]);
  }

  template<class Index>
  __device__
  void operator()(Index idx, int& past_parameter)
  {
    printf("CUDA agent {%d, %d} sees past_parameter %d\n", idx[0], idx[1], past_parameter);
  }
};

int main()
{
  test_host<agency::sequential_executor>(10);
  test_host<agency::parallel_executor>(10);
  test_host<agency::concurrent_executor>(10);

  {
    using executor_type = agency::nested_executor<agency::concurrent_executor,agency::sequential_executor>;
    test_host<executor_type>(executor_type::shape_type(4,4));
  }

  {
    using executor_type = agency::cuda::grid_executor;
    using traits = agency::executor_traits<executor_type>;
    using shape_type = traits::shape_type;

    executor_type exec;

    auto f1 = traits::make_ready_future(exec, 13);

    auto f2 = traits::then_execute(exec, f1, device_functor(), shape_type(1,1));

    f2.wait();
  }



  return 0;
}

