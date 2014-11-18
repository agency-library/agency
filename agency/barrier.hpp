#pragma once

#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

// XXX probably this needn't be exposed in a public namespace

namespace agency
{


class barrier
{
  public:
    inline explicit barrier(size_t num_threads)
      : barrier(num_threads, std::function<size_t()>([=](){return num_threads;}))
    {
      if(num_threads == 0) throw std::invalid_argument("barrier: num_threads may not be 0.");
    }

    inline barrier(size_t num_threads, size_t(*completion)())
      : barrier(num_threads, std::function<size_t()>(completion))
    {
      if(num_threads == 0) throw std::invalid_argument("barrier: num_threads may not be 0.");
    }

    inline barrier(size_t num_threads, std::function<size_t()> completion)
      : completion_(completion),
        count_(num_threads)
    {
      if(num_threads == 0) throw std::invalid_argument("barrier: num_threads may not be 0.");
    }

    inline void count_down_and_wait()
    {
      if(--count_ == 0)
      {
        // call the completion function
        size_t new_count = completion_();
        if(new_count == 0) throw std::logic_error("barrier: completion function may not return 0.");

        count_ = new_count;

        // wake everyone up
        cv_.notify_all();
      }
      else
      {
        // sleep until woken
        std::unique_lock<std::mutex> lock(mutex_);

        // XXX the lambda needs to go inside the wait() invocation
        //     instead of as an argument the stupid comma operator
        // XXX unfortunately, the barrier doesn't work when this code is written as intended
        //     we need to throw out this barrier implementation ASAP and get a better one from Olivier
        cv_.wait(lock), [this](){ this->count_ == 0; };
      }
    }

  private:
    std::function<size_t()> completion_;
    std::atomic_size_t      count_;
    std::mutex              mutex_;
    std::condition_variable cv_;
};


} // end agency

