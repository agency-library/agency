#include <agency/execution/executor/detail/new_executor_traits.hpp>
#include <type_traits>
#include <iostream>

// falls into no categories
struct not_an_executor {};


// these fall into one category
struct bulk_synchronous_executor
{
  template<class Function, class ResultFactory, class SharedFactory>
  typename std::result_of<ResultFactory(size_t)>::type
  bulk_execute(Function f, size_t n, ResultFactory result_factory, SharedFactory shared_factory);
};

struct bulk_asynchronous_executor
{
  template<class Function, class ResultFactory, class SharedFactory>
  std::future<
    typename std::result_of<ResultFactory(size_t)>::type
  >
  bulk_async_execute(Function f, size_t n, ResultFactory result_factory, SharedFactory shared_factory);
};

struct bulk_continuation_executor
{
  template<class Function, class Future, class ResultFactory, class SharedFactory>
  std::future<
    typename std::result_of<ResultFactory(size_t)>::type
  >
  bulk_then_execute(Function f, size_t n, Future& predecessor, ResultFactory result_factory, SharedFactory shared_factory);
};


// these fall into two categories
struct not_a_bulk_synchronous_executor : bulk_asynchronous_executor, bulk_continuation_executor {};
struct not_a_bulk_asynchronous_executor : bulk_synchronous_executor, bulk_continuation_executor {};
struct not_a_bulk_continuation_executor : bulk_synchronous_executor, bulk_asynchronous_executor {};


// this falls into three categories
struct complete_bulk_executor : bulk_synchronous_executor, bulk_asynchronous_executor, bulk_continuation_executor {};



int main()
{
  using namespace agency::detail::new_executor_traits_detail;

  // test not_an_executor
  static_assert(!is_bulk_executor<not_an_executor>::value, "not_an_executor is not supposed to be a bulk executor");
  static_assert(!is_bulk_synchronous_executor<not_an_executor>::value, "not_an_executor is not supposed to be a bulk synchronous executor");
  static_assert(!is_bulk_asynchronous_executor<not_an_executor>::value, "not_an_executor is not supposed to be a bulk asynchronous executor");
  static_assert(!is_bulk_continuation_executor<not_an_executor>::value, "not_an_executor is not supposed to be a bulk continuation executor");


  // test bulk_synchronous_executor
  static_assert(is_bulk_executor<bulk_synchronous_executor>::value, "bulk_synchronous_executor is supposed to be a bulk executor");
  static_assert(is_bulk_synchronous_executor<bulk_synchronous_executor>::value, "bulk_synchronous_executor is supposed to be a bulk synchronous executor");
  static_assert(!is_bulk_asynchronous_executor<bulk_synchronous_executor>::value, "bulk_synchronous_executor is not supposed to be a bulk asynchronous executor");
  static_assert(!is_bulk_continuation_executor<bulk_synchronous_executor>::value, "bulk_synchronous_executor is not supposed to be a bulk continuation executor");

  // test bulk_asynchronous_executor
  static_assert(is_bulk_executor<bulk_asynchronous_executor>::value, "bulk_asynchronous_executor is supposed to be a bulk executor");
  static_assert(!is_bulk_synchronous_executor<bulk_asynchronous_executor>::value, "bulk_asynchronous_executor is not supposed to be a bulk synchronous executor");
  static_assert(is_bulk_asynchronous_executor<bulk_asynchronous_executor>::value, "bulk_asynchronous_executor is supposed to be a bulk asynchronous executor");
  static_assert(!is_bulk_continuation_executor<bulk_asynchronous_executor>::value, "bulk_asynchronous_executor is not supposed to be a bulk continuation executor");

  // test bulk_continuation_executor
  static_assert(is_bulk_executor<bulk_continuation_executor>::value, "bulk_continuation_executor is supposed to be a bulk executor");
  static_assert(!is_bulk_synchronous_executor<bulk_continuation_executor>::value, "bulk_continuation_executor is not supposed to be a bulk synchronous executor");
  static_assert(!is_bulk_asynchronous_executor<bulk_continuation_executor>::value, "bulk_continuation_executor is not supposed to be a bulk asynchronous executor");
  static_assert(is_bulk_continuation_executor<bulk_continuation_executor>::value, "bulk_continuation_executor is supposed to be a bulk continuation executor");


  // test not_a_bulk_synchronous_executor
  static_assert(is_bulk_executor<not_a_bulk_synchronous_executor>::value,              "not_a_bulk_synchronous_executor is supposed to be a bulk executor");
  static_assert(!is_bulk_synchronous_executor<not_a_bulk_synchronous_executor>::value, "not_a_bulk_synchronous_executor is not supposed to be a bulk synchronous executor");
  static_assert(is_bulk_asynchronous_executor<not_a_bulk_synchronous_executor>::value, "not_a_bulk_synchronous_executor is supposed to be a bulk asynchronous executor");
  static_assert(is_bulk_continuation_executor<not_a_bulk_synchronous_executor>::value, "not_a_bulk_synchronous_executor is supposed to be a bulk continuation executor");

  // test not_a_bulk_asynchronous_executor
  static_assert(is_bulk_executor<not_a_bulk_asynchronous_executor>::value,               "not_a_bulk_asynchronous_executor is supposed to be a bulk executor");
  static_assert(is_bulk_synchronous_executor<not_a_bulk_asynchronous_executor>::value,   "not_a_bulk_asynchronous_executor is supposed to be a bulk synchronous executor");
  static_assert(!is_bulk_asynchronous_executor<not_a_bulk_asynchronous_executor>::value, "not_a_bulk_asynchronous_executor is not supposed to be a bulk asynchronous executor");
  static_assert(is_bulk_continuation_executor<not_a_bulk_asynchronous_executor>::value,  "not_a_bulk_asynchronous_executor is supposed to be a bulk continuation executor");

  // test not_a_bulk_continuation_executor
  static_assert(is_bulk_executor<not_a_bulk_continuation_executor>::value,               "not_a_bulk_continuation_executor is supposed to be a bulk executor");
  static_assert(is_bulk_synchronous_executor<not_a_bulk_continuation_executor>::value,   "not_a_bulk_continuation_executor is supposed to be a bulk synchronous executor");
  static_assert(is_bulk_asynchronous_executor<not_a_bulk_continuation_executor>::value,  "not_a_bulk_continuation_executor is not supposed to be a bulk asynchronous executor");
  static_assert(!is_bulk_continuation_executor<not_a_bulk_continuation_executor>::value, "not_a_bulk_continuation_executor is supposed to be a bulk continuation executor");


  // test not_a_bulk_synchronous_executor
  static_assert(is_bulk_executor<complete_bulk_executor>::value,              "complete_bulk_executor is supposed to be a bulk executor");
  static_assert(is_bulk_executor<complete_bulk_executor>::value,              "complete_bulk_executor is supposed to be a bulk synchronous executor");
  static_assert(is_bulk_asynchronous_executor<complete_bulk_executor>::value, "complete_bulk_executor is supposed to be a bulk asynchronous executor");
  static_assert(is_bulk_continuation_executor<complete_bulk_executor>::value, "complete_bulk_executor is supposed to be a bulk continuation executor");
  
  std::cout << "OK" << std::endl;

  return 0;
}

