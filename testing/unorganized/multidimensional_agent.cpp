#include <agency/agency.hpp>
#include <iostream>

int main()
{
  auto exec = agency::par2d({0,0}, {2,2});

  agency::bulk_invoke(exec, [](agency::parallel_agent_2d& self)
  {
    std::cout << "hello world from agent " << self.index() << std::endl;
  });

  return 0;
}

