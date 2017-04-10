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
#include <memory>
#include <future>
#include <unordered_map>


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
      //Number the threads 1 to number of threads, 0 is left for any thread outside of these threads
      //which would be the master thread
      for(size_t i = 0; i < num_threads; ++i){
        thread_map_[threads_[i].get_id()] = i+1;
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

    //public function used to get the thread number
    inline int get_thread_num(){
      auto thread_num_iter = thread_map_.find(std::this_thread::get_id());

      //Check to see if this is the master thread
      if(thread_num_iter == thread_map_.end())
      {
        return 0;
      }
      else
      {
        return thread_num_iter->second;
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

    //Add a private data member which gives us a mapping from thread id to an integer of size 
    //0 - max_number of threads
    std::unordered_map<std::thread::id, int> thread_map_;
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

    template<class Function, class ResultFactory, class SharedFactory>
    result_of_t<ResultFactory()>
      bulk_sync_execute(Function f, size_t n, ResultFactory result_factory, SharedFactory shared_factory)
    {
      auto result = result_factory();
      auto shared_arg = shared_factory();

      // XXX we might prefer to unconditionally execute task 0 inline
      if(n <= 1)
      {
        if(n == 1) f(0, result, shared_arg);
      }
      else
      {
        agency::detail::latch work_remaining(n);

        for(size_t idx = 0; idx < n; ++idx)
        {
          system_thread_pool().submit([=,&result,&shared_arg,&work_remaining] () mutable
          {
            f(idx, result, shared_arg);

            work_remaining.count_down(1);
          });
        }

        // wait for all the work to complete
        work_remaining.wait();
      }

      return std::move(result);
    }

  private:
    // this deleter fulfills a promise just before
    // it deletes its argument
    template<class ResultType>
    struct fulfill_promise_and_delete
    {
      std::shared_ptr<std::promise<ResultType>> shared_promise_ptr;

      void operator()(ResultType* ptr_to_result)
      {
        // move the result object into the promise
        shared_promise_ptr->set_value(std::move(*ptr_to_result));

        // delete the pointer
        delete ptr_to_result;
      }
    };
    

  public:
    // this is the overload of bulk_then_execute for non-void Future
    template<class Function, class Future, class ResultFactory, class SharedFactory,
             __AGENCY_REQUIRES(!std::is_void<future_value_t<Future>>::value)
            >
    std::future<
      result_of_t<ResultFactory()>
    >
      bulk_then_execute(Function f, size_t n, Future& predecessor, ResultFactory result_factory, SharedFactory shared_factory)
    {
      using result_type = result_of_t<ResultFactory()>;

      // create a shared promise to fulfill the result
      auto shared_promise_ptr = std::make_shared<std::promise<result_type>>();

      // get the shared promise's future
      auto result_future = shared_promise_ptr->get_future();

      // create a deleter which fulfills the promise with the result and then deletes the result
      fulfill_promise_and_delete<result_type> deleter{std::move(shared_promise_ptr)};

      // create the shared state for the result
      // note that we use our special deleter with this state
      auto shared_result_ptr = std::shared_ptr<result_type>(new result_type(result_factory()), std::move(deleter));

      // create the shared state for the shared parameter
      using shared_arg_type = result_of_t<SharedFactory()>;
      auto shared_arg_ptr = std::make_shared<shared_arg_type>(shared_factory());

      // share the incoming future
      auto shared_predecessor = future_traits<Future>::share(predecessor);

      // submit n tasks to the thread pool
      for(size_t idx = 0; idx < n; ++idx)
      {
        system_thread_pool().submit([=]() mutable
        {
          // get the predecessor future's value
          using predecessor_type = future_value_t<Future>;
          predecessor_type& predecessor_arg = const_cast<predecessor_type&>(shared_predecessor.get());

          // call the user's function
          f(idx, predecessor_arg, *shared_result_ptr, *shared_arg_ptr);

          // we explicitly release shared_result_ptr because even though this
          // lambda's invocation is complete, the lambda's lifetime
          // (and therefore shared_result_ptr's lifetime) is not necessarily complete
          // this .reset() is what fulfills the promise via shared_result_ptr's deleter
          shared_result_ptr.reset();
        });
      }

      // return the result future
      return std::move(result_future);
    }


    // this is the overload of bulk_then_execute for void Future
    template<class Function, class Future, class ResultFactory, class SharedFactory,
             __AGENCY_REQUIRES(std::is_void<future_value_t<Future>>::value)
            >
    std::future<
      result_of_t<ResultFactory()>
    >
      bulk_then_execute(Function f, size_t n, Future& predecessor, ResultFactory result_factory, SharedFactory shared_factory)
    {
      using result_type = result_of_t<ResultFactory()>;

      // create a shared promise to fulfill the result
      auto shared_promise_ptr = std::make_shared<std::promise<result_type>>();

      // get the shared promise's future
      auto result_future = shared_promise_ptr->get_future();

      // create a deleter which fulfills the promise with the result and then deletes the result
      fulfill_promise_and_delete<result_type> deleter{std::move(shared_promise_ptr)};

      // create the shared state for the result
      auto shared_result_ptr = std::shared_ptr<result_type>(new result_type(result_factory()), std::move(deleter));

      // create the shared state for the shared parameter
      using shared_arg_type = result_of_t<SharedFactory()>;
      auto shared_arg_ptr = std::make_shared<shared_arg_type>(shared_factory());

      // share the incoming future
      auto shared_predecessor = future_traits<Future>::share(predecessor);

      // submit n tasks to the thread pool
      for(size_t idx = 0; idx < n; ++idx)
      {
        system_thread_pool().submit([=]() mutable
        {
          // wait on the predecessor future
          shared_predecessor.wait();

          // call the user's function
          f(idx, *shared_result_ptr, *shared_arg_ptr);

          // we explicitly release shared_result_ptr because even though this
          // lambda's invocation is complete, the lambda's lifetime
          // (and therefore shared_result_ptr's lifetime) is not necessarily complete
          // this .reset() is what fulfills the promise via shared_result_ptr's deleter
          shared_result_ptr.reset();
        });
      }

      // return the result future
      return std::move(result_future);
    }

    size_t unit_shape() const
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

