#include <nested_executor>
#include <concurrent_executor>
#include <sequential_executor>
#include <iostream>
#include <thread>

std::mutex mut;

int main()
{
  std::nested_executor<std::concurrent_executor, std::sequential_executor> ex;

  bulk_async(ex, std::make_pair(2,2), [](std::tuple<size_t,size_t> idx)
  {
    mut.lock();
    std::cout << "Hello world from thread " << std::this_thread::get_id() << std::endl;
    std::cout << "Hello world from index " << std::get<0>(idx) << ", " << std::get<1>(idx) << std::endl;
    mut.unlock();
  }).wait();

  std::cout << std::endl;

  // test with shared variables
  ex.bulk_async([](std::tuple<size_t,size_t> idx, std::tuple<int&,int&> shared)
  {
    mut.lock();
    if(std::get<0>(idx) == 0 && std::get<1>(idx) == 0)
    {
      // set the first shared variable to 7
      std::get<0>(shared) = 7;
    }
    mut.unlock();

    mut.lock();
    std::cout << "Hello world from thread " << std::this_thread::get_id() << std::endl;
    std::cout << "Hello world from index " << std::get<0>(idx) << ", " << std::get<1>(idx) << std::endl;
    std::cout << "1st shared variable: " << std::get<0>(shared) << std::endl;
    std::cout << "2nd shared variable: " << std::get<1>(shared) << std::endl;
    mut.unlock();

    // increment 2nd shared variable
    ++std::get<1>(shared);
  },
  std::make_pair(2,2),
  std::make_tuple(13,0)).wait();

  std::cout << std::endl;

  // test with 3-deep nesting
  std::nested_executor<std::sequential_executor, std::nested_executor<std::sequential_executor,std::sequential_executor>> ex2;

  ex2.bulk_async([](std::tuple<size_t,size_t,size_t> idx)
  {
    std::cout << "(" << std::get<0>(idx) << ", " << std::get<1>(idx) << ", " << std::get<2>(idx) << ")" << std::endl;
  },
  std::make_tuple(2,2,2)
  ).wait();

  std::cout << std::endl;

  // test with shared variables
  ex2.bulk_async([](std::tuple<size_t,size_t,size_t> idx, std::tuple<int&,int&,int&> shared)
  {
    std::cout << "(" << std::get<0>(idx) << ", " << std::get<1>(idx) << ", " << std::get<2>(idx) << ")" << std::endl;
    std::cout << "1st shared variable: " << std::get<0>(shared) << std::endl;
    std::cout << "2nd shared variable: " << std::get<1>(shared) << std::endl;
    std::cout << "3rd shared variable: " << std::get<2>(shared) << std::endl;

    // increment shared variables
    std::get<0>(shared) += 1;
    std::get<1>(shared) += 2;
    std::get<2>(shared) += 3;
  },
  std::make_tuple(2,2,2),
  std::make_tuple(13,0,7)
  ).wait();


  return 0;
}

