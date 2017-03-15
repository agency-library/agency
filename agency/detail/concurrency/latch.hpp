#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/concurrency/synchronic>

#include <atomic>
#include <mutex>
#include <condition_variable>


namespace agency
{
namespace detail
{


class synchronic_latch
{
  public:
    inline explicit synchronic_latch(ptrdiff_t count)
      : counter_(count),
        released_(false)
    {
      if(counter_ == 0) throw std::invalid_argument("latch: count may not be 0.");
    }

    inline void count_down(ptrdiff_t n)
    {
      if((counter_ -= n) == 0)
      {
        notifier_.notify_all(released_, true);
      }
    }

    inline void count_down_and_wait()
    {
      count_down(1);
      wait();
    }

    inline void wait()
    {
      if(!is_ready())
      {
        notifier_.wait(released_, true);
      }
    }

    inline bool is_ready() const
    {
      return counter_ == 0;
    }

  private:
    std::atomic<size_t> counter_;
    std::atomic<bool> released_;
    std::experimental::synchronic<bool, std::experimental::synchronic_option::optimize_for_short_wait> notifier_;
};


class condition_variable_latch
{
  public:
    inline explicit condition_variable_latch(ptrdiff_t count)
      : counter_(count)
    {
      if(counter_ == 0) throw std::invalid_argument("latch: count may not be 0.");
    }

    inline void count_down(ptrdiff_t n)
    {
      std::unique_lock<std::mutex> lock(mutex_);

      counter_ -= n;

      if(unsafe_is_ready())
      {
        // unblock all blocking threads
        cv_.notify_all();
      }
    }

    inline void count_down_and_wait()
    {
      count_down(1);
      wait();
    }

    inline void wait()
    {
      std::unique_lock<std::mutex> lock(mutex_);

      if(!unsafe_is_ready())
      {
        cv_.wait(lock, [=]{ return this->unsafe_is_ready(); });
      }
    }

    inline bool is_ready() const
    {
      std::unique_lock<std::mutex> lock(mutex_);
      return unsafe_is_ready();
    }

  private:
    inline bool unsafe_is_ready() const
    {
      return counter_ == 0;
    }

    size_t                  counter_;
    mutable std::mutex      mutex_;
    std::condition_variable cv_;
};


// condition_variable_latch is much faster than synchronic_latch for some reason
using latch = condition_variable_latch;


} // end detail
} // end agency

