#include <iostream>

#include <agency/execution/executor/executor_traits.hpp>
#include "../test_executors.hpp"


struct bulk_executor_without_category
{
  template<class Function, class ResultFactory, class SharedFactory>
  typename std::result_of<ResultFactory()>::type
  bulk_execute(Function f, size_t n, ResultFactory result_factory, SharedFactory shared_factory);
};

struct bulk_executor_with_category
{
  using execution_category = agency::sequenced_execution_tag;

  template<class Function, class ResultFactory, class SharedFactory>
  typename std::result_of<ResultFactory()>::type
  bulk_execute(Function f, size_t n, ResultFactory result_factory, SharedFactory shared_factory);
};


int main()
{
  using namespace agency;

  static_assert(!agency::detail::is_detected<executor_execution_category_t, not_an_executor>::value,
    "executor_execution_category_t<not_an_executor> should not be detected");

  static_assert(agency::detail::is_detected_exact<unsequenced_execution_tag, executor_execution_category_t, bulk_executor_without_category>::value,
    "bulk_executor_without_category should have unsequenced_execution_tag execution_category");

  static_assert(agency::detail::is_detected_exact<sequenced_execution_tag, executor_execution_category_t, bulk_executor_with_category>::value,
    "bulk_executor_with_category should have sequenced_execution_tag execution_category");

  std::cout << "OK" << std::endl;

  return 0;
}

