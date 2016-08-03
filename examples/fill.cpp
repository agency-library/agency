#include <agency/agency.hpp>
#include <vector>
#include <cassert>
#include <iostream>

int main()
{
  // create an array
  std::vector<int> array(100);

  // create a group of parallel agents
  agency::bulk_invoke(agency::par(array.size()), [&array](agency::parallel_agent& self)
  {
    // set the value of the agent's corresponding array element
    array[self.index()] = 42;
  });

  // check the result
  assert(array == std::vector<int>(100,42));

  std::cout << "OK" << std::endl;

  return 0;
}

