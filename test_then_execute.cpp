#include <agency/executor_traits.hpp>
#include <agency/sequential_executor.hpp>
#include <agency/concurrent_executor.hpp>
#include <agency/parallel_executor.hpp>
#include <agency/nested_executor.hpp>
#include <mutex>
#include <thread>
#include <cassert>

template<class Executor, class Shape>
void test(Shape shape)
{
  using executor_type = Executor;
  using traits = agency::executor_traits<executor_type>;
  using index_type = typename traits::index_type;
  
  executor_type exec;
  
  {
    auto f1 = traits::template make_ready_future<void>(exec);
    
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
    auto f1 = traits::template make_ready_future<int>(exec, 13);

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

int main()
{
  test<agency::sequential_executor>(10);
  test<agency::parallel_executor>(10);
  test<agency::concurrent_executor>(10);

  using executor_type = agency::nested_executor<agency::concurrent_executor,agency::sequential_executor>;
  test<executor_type>(executor_type::shape_type(4,4));

  return 0;
}

