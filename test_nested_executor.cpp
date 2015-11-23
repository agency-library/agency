#include <agency/nested_executor.hpp>
#include <agency/concurrent_executor.hpp>
#include <agency/sequential_executor.hpp>
#include <agency/bulk_invoke.hpp>
#include <iostream>
#include <thread>

std::mutex mut;

int main()
{
  agency::nested_executor<agency::concurrent_executor, agency::sequential_executor> ex;

  agency::bulk_async(ex, std::make_pair(2,2), [](std::tuple<size_t,size_t> idx)
  {
    mut.lock();
    std::cout << "Hello world from thread " << std::this_thread::get_id() << std::endl;
    std::cout << "Hello world from index " << std::get<0>(idx) << ", " << std::get<1>(idx) << std::endl;
    mut.unlock();
  }).wait();

  std::cout << std::endl;

//  // test with shared variables
//  ex.async_execute([](std::tuple<size_t,size_t> idx, int& shared0, int& shared1)
//  {
//    mut.lock();
//    if(std::get<0>(idx) == 0 && std::get<1>(idx) == 0)
//    {
//      // set the first shared variable to 7
//      shared0 = 7;
//    }
//    mut.unlock();
//
//    mut.lock();
//    std::cout << "Hello world from thread " << std::this_thread::get_id() << std::endl;
//    std::cout << "Hello world from index " << std::get<0>(idx) << ", " << std::get<1>(idx) << std::endl;
//    std::cout << "1st shared variable: " << shared0 << std::endl;
//    std::cout << "2nd shared variable: " << shared1 << std::endl;
//    mut.unlock();
//
//    // increment 2nd shared variable
//    ++shared1;
//  },
//  std::make_pair(2,2),
//  13, 0
//  ).wait();
//
//  std::cout << std::endl;
//
//  // test with 3-deep nesting
//  agency::nested_executor<agency::sequential_executor, agency::nested_executor<agency::sequential_executor,agency::sequential_executor>> ex2;
//
//  ex2.async_execute([](std::tuple<size_t,size_t,size_t> idx)
//  {
//    std::cout << "(" << std::get<0>(idx) << ", " << std::get<1>(idx) << ", " << std::get<2>(idx) << ")" << std::endl;
//  },
//  std::make_tuple(2,2,2)
//  ).wait();
//
//  std::cout << std::endl;
//
//  // test with shared variables
//  ex2.async_execute([](std::tuple<size_t,size_t,size_t> idx, int& shared0, int& shared1, int& shared2)
//  {
//    std::cout << "(" << std::get<0>(idx) << ", " << std::get<1>(idx) << ", " << std::get<2>(idx) << ")" << std::endl;
//    std::cout << "1st shared variable: " << shared0 << std::endl;
//    std::cout << "2nd shared variable: " << shared1 << std::endl;
//    std::cout << "3rd shared variable: " << shared2 << std::endl;
//
//    // increment shared variables
//    shared0 += 1;
//    shared1 += 2;
//    shared2 += 3;
//  },
//  std::make_tuple(2,2,2),
//  13, 0, 7
//  ).wait();


  return 0;
}

