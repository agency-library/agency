#pragma once

#include <agency/detail/config.hpp>

#include <atomic>
#include <mutex>
#include <condition_variable>

#if __cplusplus > 201103L
#include <agency/detail/concurrency/synchronic>
#endif


namespace agency
{
namespace detail
{

#if __cplusplus > 201103L

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
      if(--counter_ == 0)
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

#endif // c++14


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

      if(is_ready())
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
      if(!is_ready())
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [=]{ return this->is_ready(); });
      }
    }

    inline bool is_ready() const
    {
      return counter_ == 0;
    }

  private:
    size_t                  counter_;
    std::mutex              mutex_;
    std::condition_variable cv_;
};


using latch = condition_variable_latch;


} // end detail
} // end agency

