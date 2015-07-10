#include <agency/sequential_executor.hpp>
#include <agency/concurrent_executor.hpp>
#include <agency/parallel_executor.hpp>
#include <agency/nested_executor.hpp>
#include <agency/executor_traits.hpp>
#include <iostream>

int main()
{
  std::cout << "is_executor<agency::sequential_executor>: " << agency::is_executor<agency::sequential_executor>::value << std::endl;
  std::cout << "has_then_execute<agency::sequential_executor>: " << agency::detail::has_multi_agent_then_execute<agency::sequential_executor>::value << std::endl;

  std::cout << "is_executor<agency::concurrent_executor>: " << agency::is_executor<agency::concurrent_executor>::value << std::endl;
  std::cout << "has_then_execute<agency::concurrent_executor>: " << agency::detail::has_multi_agent_then_execute<agency::concurrent_executor>::value << std::endl;

  std::cout << "is_executor<agency::parallel_executor>: " << agency::is_executor<agency::parallel_executor>::value << std::endl;
  std::cout << "has_then_execute<agency::parallel_executor>: " << agency::detail::has_multi_agent_then_execute<agency::parallel_executor>::value << std::endl;

  std::cout << "is_executor<agency::nested_executor<agency::concurrent_executor,agency::sequential_executor>>: " << agency::is_executor<agency::nested_executor<agency::concurrent_executor,agency::sequential_executor>>::value << std::endl;
  std::cout << "has_then_execute<agency::nested_executor<agency::concurrent_executor,agency::sequential_executor>>: " << agency::detail::has_multi_agent_then_execute<agency::nested_executor<agency::concurrent_executor,agency::sequential_executor>>::value << std::endl;

  return 0;
}

