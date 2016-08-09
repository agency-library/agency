#include <agency/scoped_executor.hpp>
#include <agency/concurrent_executor.hpp>
#include <agency/sequenced_executor.hpp>
#include <agency/bulk_invoke.hpp>
#include <iostream>
#include <thread>
#include <tuple>

std::mutex mut;

int main()
{
  // XXX we should really assert that the results match the expected value

  {
    using executor_type = agency::scoped_executor<agency::concurrent_executor, agency::sequenced_executor>;
    executor_type ex;

    using traits = agency::executor_traits<executor_type>;

    using index_type = traits::index_type;
    using shape_type = traits::shape_type;

    traits::async_execute(ex, 
      [](index_type idx)
      {
        mut.lock();
        std::cout << "Hello world from thread " << std::this_thread::get_id() << std::endl;
        std::cout << "Hello world from index " << std::get<0>(idx) << ", " << std::get<1>(idx) << std::endl;
        mut.unlock();
      },
      shape_type(2,2)
    ).wait();

    std::cout << std::endl;

    // test with shared variables
    traits::async_execute(ex,
      [](index_type idx, int& shared0, int& shared1)
      {
        mut.lock();
        if(std::get<0>(idx) == 0 && std::get<1>(idx) == 0)
        {
          // set the first shared variable to 7
          shared0 = 7;
        }
        mut.unlock();

        mut.lock();
        std::cout << "Hello world from thread " << std::this_thread::get_id() << std::endl;
        std::cout << "Hello world from index " << std::get<0>(idx) << ", " << std::get<1>(idx) << std::endl;
        std::cout << "1st shared variable: " << shared0 << std::endl;
        std::cout << "2nd shared variable: " << shared1 << std::endl;
        mut.unlock();

        // increment 2nd shared variable
        ++shared1;
      },
      shape_type(2,2),
      []{ return 13; },
      []{ return 0; }
    ).wait();

    std::cout << std::endl;
  }

  {
    // test with 3-deep nesting
    
    using executor_type = agency::scoped_executor<agency::sequenced_executor, agency::scoped_executor<agency::sequenced_executor,agency::sequenced_executor>>;
    using traits = agency::executor_traits<executor_type>;
    executor_type ex;

    using index_type = traits::index_type;
    using shape_type = traits::shape_type;

    traits::async_execute(ex,
      [](index_type idx)
      {
        std::cout << "(" << std::get<0>(idx) << ", " << std::get<1>(idx) << ", " << std::get<2>(idx) << ")" << std::endl;
      },
      shape_type(2,2,2)
    ).wait();

    std::cout << std::endl;

    // test with shared variables
    traits::async_execute(ex,
      [](index_type idx, int& shared0, int& shared1, int& shared2)
      {
        std::cout << "(" << std::get<0>(idx) << ", " << std::get<1>(idx) << ", " << std::get<2>(idx) << ")" << std::endl;
        std::cout << "1st shared variable: " << shared0 << std::endl;
        std::cout << "2nd shared variable: " << shared1 << std::endl;
        std::cout << "3rd shared variable: " << shared2 << std::endl;

        // increment shared variables
        shared0 += 1;
        shared1 += 2;
        shared2 += 3;
      },
      shape_type(2,2,2),
      []{ return 13; },
      []{ return 0; },
      []{ return 7; }
    ).wait();
  }

  return 0;
}

