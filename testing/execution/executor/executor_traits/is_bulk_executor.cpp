#include <agency/execution/executor/executor_traits.hpp>
#include <type_traits>
#include <iostream>

#include "../test_executors.hpp"


int main()
{
  using namespace agency;

  // test not_an_executor
  static_assert(!is_bulk_executor<not_an_executor>::value, "not_an_executor is not supposed to be a bulk executor");
  static_assert(!detail::is_bulk_twoway_executor<not_an_executor>::value, "not_an_executor is not supposed to be a bulk twoway executor");
  static_assert(!is_bulk_continuation_executor<not_an_executor>::value, "not_an_executor is not supposed to be a bulk continuation executor");

  // test bulk_twoway_executor
  static_assert(is_bulk_executor<bulk_twoway_executor>::value, "bulk_twoway_executor is supposed to be a bulk executor");
  static_assert(detail::is_bulk_twoway_executor<bulk_twoway_executor>::value, "bulk_twoway_executor is supposed to be a bulk twoway executor");
  static_assert(!is_bulk_continuation_executor<bulk_twoway_executor>::value, "bulk_twoway_executor is not supposed to be a bulk continuation executor");

  // test bulk_continuation_executor
  static_assert(is_bulk_executor<bulk_continuation_executor>::value, "bulk_continuation_executor is supposed to be a bulk executor");
  static_assert(!detail::is_bulk_twoway_executor<bulk_continuation_executor>::value, "bulk_continuation_executor is not supposed to be a bulk twoway executor");
  static_assert(is_bulk_continuation_executor<bulk_continuation_executor>::value, "bulk_continuation_executor is supposed to be a bulk continuation executor");

  // test not_a_bulk_twoway_executor
  static_assert(is_bulk_executor<not_a_bulk_twoway_executor>::value,               "not_a_bulk_twoway_executor is supposed to be a bulk executor");
  static_assert(!detail::is_bulk_twoway_executor<not_a_bulk_twoway_executor>::value, "not_a_bulk_twoway_executor is not supposed to be a bulk twoway executor");
  static_assert(is_bulk_continuation_executor<not_a_bulk_twoway_executor>::value,  "not_a_bulk_twoway_executor is supposed to be a bulk continuation executor");

  // test not_a_bulk_continuation_executor
  static_assert(is_bulk_executor<not_a_bulk_continuation_executor>::value,               "not_a_bulk_continuation_executor is supposed to be a bulk executor");
  static_assert(detail::is_bulk_twoway_executor<not_a_bulk_continuation_executor>::value,  "not_a_bulk_continuation_executor is not supposed to be a bulk twoway executor");
  static_assert(!is_bulk_continuation_executor<not_a_bulk_continuation_executor>::value, "not_a_bulk_continuation_executor is supposed to be a bulk continuation executor");

  // test not_a_bulk_synchronous_executor
  static_assert(is_bulk_executor<complete_bulk_executor>::value,              "complete_bulk_executor is supposed to be a bulk executor");
  static_assert(detail::is_bulk_twoway_executor<complete_bulk_executor>::value, "complete_bulk_executor is supposed to be a bulk twoway executor");
  static_assert(is_bulk_continuation_executor<complete_bulk_executor>::value, "complete_bulk_executor is supposed to be a bulk continuation executor");
  
  std::cout << "OK" << std::endl;

  return 0;
}

