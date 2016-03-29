#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution_categories.hpp>
#include <agency/parallel_executor.hpp>
#include <agency/vector_executor.hpp>
#include <agency/nested_executor.hpp>
#include <agency/flattened_executor.hpp>
#include <agency/detail/concurrency/latch.hpp>
#include <agency/detail/concurrency/concurrent_queue.hpp>

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
             class = typename std::result_of<Function()>::type>
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

  private:
    inline void work()
    {
      std::function<void()> task;

      while(tasks_.wait_and_pop(task))
      {
        task();
      }
    }

    agency::detail::concurrent_queue<std::function<void()>> tasks_;
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
    // XXX should really implement then_execute(), but we'll start with execute() for now
    template<class Factory1, class Function, class Factory2>
    typename std::result_of<Factory1(size_t)>::type
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
                 typename std::result_of<Function(size_t, typename std::result_of<Factory()>::type&)>::type
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

    size_t shape() const
    {
      return system_thread_pool().size();
    }
};


// compose thread_pool_executor with other fancy executors
// to yield a parallel_thread_pool_executor
using parallel_thread_pool_executor = agency::flattened_executor<
  agency::nested_executor<
    thread_pool_executor,
    agency::this_thread::parallel_executor
  >
>;


// compose thread_pool_executor with other fancy executors
// to yield a parallel_vector_thread_pool_executor
using parallel_vector_thread_pool_executor = agency::flattened_executor<
  agency::nested_executor<
    thread_pool_executor,
    agency::this_thread::vector_executor
  >
>;


} // end detail
} // end agency

