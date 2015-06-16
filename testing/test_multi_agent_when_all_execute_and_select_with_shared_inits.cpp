#include <agency/future.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/coordinate.hpp>
#include <iostream>
#include <cassert>

struct my_executor {};

struct my_scoped_executor
{
  using execution_category = agency::nested_execution_tag<
    agency::sequential_execution_tag,
    agency::sequential_execution_tag
  >;

  using shape_type = agency::size2;

  template<class Function>
  struct functor
  {
    mutable Function f;
    shape_type shape;

    template<class... Args>
    void operator()(Args&... args) const
    {
      for(size_t i = 0; i < shape[0]; ++i)
      {
        for(size_t j = 0; j < shape[1]; ++j)
        {
          f(agency::size2(i,j), args...);
        }
      }
    }
  };

  template<size_t... Indices, class Function, class TupleOfFutures>
  std::future<
    agency::detail::when_all_execute_and_select_result_t<
      agency::detail::index_sequence<Indices...>,
      typename std::decay<TupleOfFutures>::type
    >
  >
    when_all_execute_and_select(Function f, shape_type shape, TupleOfFutures&& futures)
  {
    return agency::when_all_execute_and_select<Indices...>(functor<Function>{f,shape}, std::forward<TupleOfFutures>(futures));
  }
};

int main()
{
  {
    size_t n = 100;
    int addend = 13;

    auto futures = std::make_tuple(agency::detail::make_ready_future<int>(addend));

    std::atomic<int> counter(n);
    int current_sum = 0;
    int result = 0;

    std::mutex mut;
    my_executor exec;
    std::future<int> fut = agency::new_executor_traits<my_executor>::when_all_execute_and_select<0>(exec, [&](size_t idx, int& addend, int& current_sum)
    {
      mut.lock();
      current_sum += addend;
      mut.unlock();

      auto prev_counter_value = counter.fetch_sub(1);

      // the last agent stores the current_sum to the result
      if(prev_counter_value == 1)
      {
        result = current_sum;
      }
    },
    n,
    futures,
    current_sum);

    auto got = fut.get();

    assert(got == 13);
    assert(result == addend * n);
  }

  {
    my_scoped_executor exec;

    agency::size2 shape(6,6);
    int addend = 1;

    auto futures = std::make_tuple(agency::detail::make_ready_future<int>(addend));

    std::atomic<int> total_counter(shape[0]);
    std::array<std::atomic<int>, 10> group_counters;
    std::fill(group_counters.begin(), group_counters.end(), shape[1]);

    int current_total_sum = 0;
    int current_group_sum = 0;
    int result = 0;

    std::mutex mut;
    auto fut = agency::new_executor_traits<my_scoped_executor>::when_all_execute_and_select<0>(exec, [&](agency::size2 idx, int& addend, int& current_total_sum, int& current_group_sum)
    {
      mut.lock();
      current_group_sum += addend;
      mut.unlock();

      auto prev_group_counter_value = group_counters[idx[0]].fetch_sub(1);

      // the last agent in the group adds to the current total sum
      if(prev_group_counter_value == 1)
      {
        mut.lock();
        current_total_sum += current_group_sum;
        mut.unlock();

        auto prev_total_counter_value = total_counter.fetch_sub(1);

        // the last agent overall stores the current_total_sum to the result
        if(prev_total_counter_value == 1)
        {
          result = current_total_sum;
        }
      }
    },
    shape,
    futures,
    0,
    0);

    auto got = fut.get();

    assert(got == addend);
    assert(result == addend * shape[0] * shape[1]);
  }

  std::cout << "OK" << std::endl;

  return 0;
}

