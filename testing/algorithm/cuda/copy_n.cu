#include <agency/cuda/algorithm/copy_n.hpp>
#include <agency/cuda/memory/allocator/managed_allocator.hpp>
#include <agency/cuda/memory/allocator/device_allocator.hpp>
#include <agency/cuda/execution.hpp>
#include <agency/container/vector.hpp>
#include <cassert>
#include <iostream>


template<class Container, class ExecutionPolicy>
void test(ExecutionPolicy policy)
{
  Container source(policy, 10, 7);
  Container dest(policy, source.size(), 13);

  auto result = agency::cuda::copy_n(policy, source.begin(), source.size(), dest.begin());

  assert(source.end() == agency::get<0>(result));
  assert(dest.end()   == agency::get<1>(result));
  assert(Container(10, 7) == dest);
}

int main()
{
  using namespace agency;

  test<vector<int, cuda::managed_allocator<int>>>(seq);
  test<vector<int, cuda::managed_allocator<int>>>(par);
  test<vector<int, cuda::managed_allocator<int>>>(cuda::par);

  // XXX disable this while operator== used in test() above executes in the current thread
  //test<vector<int, cuda::device_allocator<int>>>(cuda::par);

  std::cout << "OK" << std::endl;
}

