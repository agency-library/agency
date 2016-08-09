#include <agency/agency.hpp>
#include <iostream>

void hello(agency::sequenced_agent& self)
{
  std::cout << "Hello, world from agent " << self.index() << std::endl;
}

int main()
{
  // create 10 sequenced_agents to execute the hello() task in bulk
  agency::bulk_invoke(agency::seq(10), hello);

  return 0;
}

