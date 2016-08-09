#include <agency/agency.hpp>
#include <iostream>

int main()
{
  using namespace agency::detail::executor_traits_detail;

  std::cout << "is_executor<agency::sequenced_executor>: " << agency::is_executor<agency::sequenced_executor>::value << std::endl;
  std::cout << "has_then_execute<agency::sequenced_executor>: " << has_any_multi_agent_then_execute<agency::sequenced_executor>::value << std::endl;

  std::cout << "is_executor<agency::concurrent_executor>: " << agency::is_executor<agency::concurrent_executor>::value << std::endl;
  std::cout << "has_then_execute<agency::concurrent_executor>: " << has_any_multi_agent_then_execute<agency::concurrent_executor>::value << std::endl;

  std::cout << "is_executor<agency::vector_executor>: " << agency::is_executor<agency::vector_executor>::value << std::endl;
  std::cout << "has_then_execute<agency::vector_executor>: " << has_any_multi_agent_then_execute<agency::vector_executor>::value << std::endl;

  std::cout << "is_executor<agency::parallel_executor>: " << agency::is_executor<agency::parallel_executor>::value << std::endl;
  std::cout << "has_then_execute<agency::parallel_executor>: " << has_any_multi_agent_then_execute<agency::parallel_executor>::value << std::endl;

  std::cout << "is_executor<agency::scoped_executor<agency::concurrent_executor,agency::sequenced_executor>>: " << agency::is_executor<agency::scoped_executor<agency::concurrent_executor,agency::sequenced_executor>>::value << std::endl;
  std::cout << "has_then_execute<agency::scoped_executor<agency::concurrent_executor,agency::sequenced_executor>>: " << has_any_multi_agent_then_execute<agency::scoped_executor<agency::concurrent_executor,agency::sequenced_executor>>::value << std::endl;

  return 0;
}

