#include <iostream>
#include <agency/execution_policy.hpp>
#include <agency/coordinate.hpp>

template<class ExecutionCategory, size_t N>
using basic_multidimensional_agent = agency::detail::basic_execution_agent<ExecutionCategory, agency::point<size_t,N>>;

using parallel_agent_2d = basic_multidimensional_agent<agency::parallel_execution_tag, 2>;

const agency::detail::basic_execution_policy<parallel_agent_2d, agency::parallel_executor> par2d{};

int main()
{
  auto exec = par2d({0,0}, {2,2});

  agency::bulk_invoke(exec, [](parallel_agent_2d& self)
  {
    std::cout << "hello world from agent " << self.index() << std::endl;
  });

  return 0;
}

