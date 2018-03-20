#include <agency/execution/execution_policy.hpp>
#include <iostream>

template<class ExecutionPolicy, class Executor>
void test(const ExecutionPolicy& policy, const Executor& ex)
{
  auto p1 = policy.on(ex);
  assert(p1.executor() == ex);
}

int main()
{
  // test seq
  test(agency::seq, agency::sequenced_executor());

  // test par
  test(agency::par, agency::sequenced_executor());
  test(agency::par, agency::parallel_executor());
  test(agency::par, agency::concurrent_executor());

  // test con
  test(agency::con, agency::concurrent_executor());

  std::cout << "OK" << std::endl;
}

