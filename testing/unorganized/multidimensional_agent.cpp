#include <iostream>
#include <agency/bulk_invoke.hpp>
#include <agency/execution_policy.hpp>
#include <agency/coordinate.hpp>

const agency::detail::basic_execution_policy<agency::parallel_agent_2d, agency::parallel_executor> par2d{};

int main()
{
  auto exec = par2d({0,0}, {2,2});

  agency::bulk_invoke(exec, [](agency::parallel_agent_2d& self)
  {
    std::cout << "hello world from agent " << self.index() << std::endl;
  });

  return 0;
}

