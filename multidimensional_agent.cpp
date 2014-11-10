#include <iostream>
#include <execution_policy>
#include <coordinate>

template<class ExecutionCategory, size_t N>
using basic_multidimensional_agent = std::__basic_execution_agent<ExecutionCategory, std::point<size_t,N>>;

using parallel_agent_2d = basic_multidimensional_agent<std::parallel_execution_tag, 2>;

const std::__basic_execution_policy<parallel_agent_2d, std::parallel_executor> par2d{};

int main()
{
  auto exec = par2d({0,0}, {2,2});

  std::bulk_invoke(exec, [](parallel_agent_2d& self)
  {
    std::cout << "hello world from agent " << self.index() << std::endl;
  });

  return 0;
}

