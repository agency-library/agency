#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/execution_categories.hpp>
#include <agency/execution/executor/parallel_executor.hpp>
#include <agency/execution/executor/vector_executor.hpp>
#include <agency/execution/executor/scoped_executor.hpp>
#include <agency/execution/executor/flattened_executor.hpp>
#include <agency/detail/concurrency/latch.hpp>
#include <agency/detail/concurrency/concurrent_queue.hpp>
#include <agency/detail/unique_function.hpp>
#include <agency/future.hpp>
#include <agency/detail/type_traits.hpp>

#include <thread>
#include <vector>
#include <algorithm>


namespace agency
{
namespace detail
{


class thread_pool
{
  private:
    struct joining_thread : std::thread
    {
      using std::thread::thread;

      joining_thread(joining_thread&&) = default;

      ~joining_thread()
      {
        if(joinable()) join();
      }
    };

  public:
    explicit thread_pool(size_t num_threads = std::max(1u, std::thread::hardware_concurrency()))
    {
      for(size_t i = 0; i < num_threads; ++i)
      {
        threads_.emplace_back([this]
        {
          work();
        });
      }
    }
    
    ~thread_pool()
    {
      tasks_.close();
      threads_.clear();
    }

    template<class Function,
             class = result_of_t<Function()>>
    inline void submit(Function&& f)
    {
      auto is_this_thread = [=](const joining_thread& t)
      {
        return t.get_id() == std::this_thread::get_id();
      };

      // guard against self-submission which may result in deadlock
      // XXX it might be faster to compare this to a thread_local variable
      if(std::find_if(threads_.begin(), threads_.end(), is_this_thread) == threads_.end())
      {
        tasks_.emplace(std::forward<Function>(f));
      }
      else
      {
        // the submitting thread is part of this pool so execute immediately 
        std::forward<Function>(f)();
      }
    }

    inline size_t size() const
    {
      return threads_.size();
    }

    template<class Function, class... Args>
    std::future<result_of_t<Function(Args...)>>
      async(Function&& f, Args&&... args)
    {
      // bind f & args together
      auto g = std::bind(std::forward<Function>(f), std::forward<Args>(args)...);

      using result_type = result_of_t<Function(Args...)>;

      // create a packaged task
      std::packaged_task<result_type()> task(std::move(g));

      // get the packaged task's future so we can return it at the end
      auto result_future = task.get_future();

      // move the packaged task into the thread pool
      submit(std::move(task));

      return std::move(result_future);
    }


  private:
    inline void work()
    {
      unique_function<void()> task;

      while(tasks_.wait_and_pop(task))
      {
        task();
      }
    }

    agency::detail::concurrent_queue<unique_function<void()>> tasks_;
    std::vector<joining_thread> threads_;
};



inline thread_pool& system_thread_pool()
{
  static thread_pool resource;
  return resource;
}


class thread_pool_executor
{
  public:
    using execution_category = parallel_execution_tag;

    template<class Factory1, class Function, class Factory2>
    result_of_t<Factory1(size_t)>
      execute(Function f, Factory1 result_factory, size_t n, Factory2 shared_factory)
    {
      auto result = result_factory(n);
      auto shared_arg = shared_factory();

      // XXX we might prefer to unconditionally execute task 0 inline
      if(n <= 1)
      {
        if(n == 1) result[0] = f(0, shared_arg);
      }
      else
      {
        agency::detail::latch work_remaining(n);

        for(size_t idx = 0; idx < n; ++idx)
        {
          system_thread_pool().submit([=,&result,&shared_arg,&work_remaining] () mutable
          {
            result[idx] = f(idx, shared_arg);

            work_remaining.count_down(1);
          });
        }

        // wait for all the work to complete
        work_remaining.wait();
      }

      return std::move(result);
    }

    template<class Function, class Factory,
             class = typename std::enable_if<
               std::is_void<
                 result_of_t<Function(size_t, result_of_t<Factory()>&)>
               >::value
             >::type>
    void execute(Function f, size_t n, Factory shared_factory)
    {
      auto shared_arg = shared_factory();

      // execute small workloads immediately
      if(n <= 1)
      {
        if(n == 1) f(0, shared_arg);
      }
      else
      {
        agency::detail::latch work_remaining(n);
  
        for(size_t idx = 0; idx < n; ++idx)
        {
          system_thread_pool().submit([=,&shared_arg,&work_remaining]() mutable
          {
            f(idx, shared_arg);
  
            work_remaining.count_down(1);
          });
        }
  
        // wait for all the work to complete
        work_remaining.wait();
      }
    }

    template<class Function>
    void execute(Function f, size_t n)
    {
      // execute small workloads immediately
      if(n <= 1)
      {
        if(n == 1) f(0);
      }
      else
      {
        agency::detail::latch work_remaining(n);
  
        for(size_t idx = 0; idx < n; ++idx)
        {
          system_thread_pool().submit([=,&work_remaining]() mutable
          {
            f(idx);
            work_remaining.count_down(1);
          });
        }
  
        // wait for all the work to complete
        work_remaining.wait();
      }
    }

    template<size_t... Indices, class Function, class... Futures>
    std::future<
      detail::when_all_and_select_result_t<
        detail::index_sequence<Indices...>, typename std::decay<Futures>::type...
      >
    >
      when_all_execute_and_select_impl_impl(index_sequence<Indices...>, Function&& f, Futures&&... futures)
    {
      return system_thread_pool().async(when_all_execute_and_select_functor<Indices...>(), std::forward<Function>(f), std::move(futures)...);
    }

    template<size_t... SelectedIndices, size_t... TupleIndices, class Function, class TupleOfFutures>
    std::future<
      detail::when_all_execute_and_select_result_t<
        index_sequence<SelectedIndices...>,
        decay_t<TupleOfFutures>
      >
    >
      when_all_execute_and_select_impl(index_sequence<SelectedIndices...> indices, index_sequence<TupleIndices...>, Function&& f, TupleOfFutures&& futures)
    {
      return when_all_execute_and_select_impl_impl(indices, std::forward<Function>(f), detail::get<TupleIndices>(std::forward<TupleOfFutures>(futures))...);
    }


    // XXX the reason we implemented single-agent when_all_execute_and_select
    //     is to fix #246 -- bulk_async(par, ...) was executing synchronously
    //     this is because:
    //       * parallel_executor is implemented by flattening an executor_array
    //       * executor_array::then_execute() is implemented with when_all_execute_and_select() on the outer executor
    //         * in this case, the outer executor is a thread_pool_executor
    //       * because thread_pool_executor did not previously have when_all_execute_and_select(), the asynchrony was created via std::async(std::launch::deferred, ...)
    // XXX what we need to do in the future is something like this:
    //       * implement a fix to #248
    //       * simplify executor_array::then_execute() to use outer_executor().bulk_then_execute()
    //       * implement bulk_execute(), bulk_async_execute(), and bulk_then_execute() for thread_pool_executor
    template<size_t... Indices, class Function, class TupleOfFutures>
    std::future<
      detail::when_all_execute_and_select_result_t<
        index_sequence<Indices...>,
        decay_t<TupleOfFutures>
      >
    >
      when_all_execute_and_select(Function&& f, TupleOfFutures&& futures)
    {
      return when_all_execute_and_select_impl(
        index_sequence<Indices...>(),
        make_tuple_indices(futures),
        std::forward<Function>(f),
        std::forward<TupleOfFutures>(futures)
      );
    }

    size_t shape() const
    {
      return system_thread_pool().size();
    }
};


// compose thread_pool_executor with other fancy executors
// to yield a parallel_thread_pool_executor
using parallel_thread_pool_executor = agency::flattened_executor<
  agency::scoped_executor<
    thread_pool_executor,
    agency::this_thread::parallel_executor
  >
>;


// compose thread_pool_executor with other fancy executors
// to yield a parallel_vector_thread_pool_executor
using parallel_vector_thread_pool_executor = agency::flattened_executor<
  agency::scoped_executor<
    thread_pool_executor,
    agency::this_thread::vector_executor
  >
>;


} // end detail
} // end agency

