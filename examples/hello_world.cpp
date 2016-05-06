#include <agency/bulk_invoke.hpp>
#include <agency/execution_policy.hpp>
#include <iostream>

void hello(agency::sequential_agent& self)
{
  std::cout << "Hello, world from agent " << self.index() << std::endl;
}

int main()
{
  // create 10 sequential_agents to execute the hello() task in bulk
  agency::bulk_invoke(agency::seq(10), hello);

  return 0;
}

