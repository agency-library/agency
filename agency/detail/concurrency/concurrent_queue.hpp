#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/concurrency/synchronic>

#include <queue>
#include <atomic>
#include <condition_variable>


namespace agency
{
namespace detail
{


// this type increments a counter when it is constructed
// and decrements it upon destruction
template<class T>
struct scope_bumper
{
  scope_bumper(std::atomic<T>& counter)
    : counter_(counter)
  {
    ++counter_;
  }

  ~scope_bumper()
  {
    --counter_;
  }

  std::atomic<T>& counter_;
};


template<class T>
void wait_until_equal(const std::atomic<T>& a, const T& value)
{
  // implement this with a spin loop
  while(a != value)
  {
    // spin
  }
}


enum queue_status
{
  open_and_empty = 0,
  open_and_ready = 1,
  closed = 2
};


template<class T>
class synchronic_concurrent_queue
{
  public:
    synchronic_concurrent_queue()
      : num_poppers_(0),
        status_(open_and_empty)
    {
    }

    ~synchronic_concurrent_queue()
    {
      close();
    }

    void close()
    {
      {
        std::unique_lock<std::mutex> lock(mutex_);

        // don't attempt to close a closed queue
        if(status_ == closed) return;

        // notify that we're closing
        notifier_.notify_all(status_, (int)closed);
      }
      
      // wait until all the poppers have finished with wait_and_pop() 
      detail::wait_until_equal(num_poppers_, 0);
    }

    bool is_closed()
    {
      std::unique_lock<std::mutex> lock(mutex_);

      return status_ == closed;
    }

    template<class... Args>
    queue_status emplace(Args&&... args)
    {
      std::unique_lock<std::mutex> lock(mutex_);

      if(status_ == closed)
      {
        return queue_status::closed;
      }

      items_.emplace(std::forward<Args>(args)...);

      notifier_.notify_one(status_, (int)open_and_ready); 

      return queue_status::open_and_ready;
    }

    queue_status push(const T& item)
    {
      return emplace(item);
    }

    // XXX this should return queue_status
    bool wait_and_pop(T& item)
    {
      scope_bumper<int> popping(num_poppers_);

      while(true)
      {
        notifier_.wait_for_change(status_, (int)open_and_empty);

        {
          std::unique_lock<std::mutex> lock(mutex_);

          // if the queue is closed, return
          if(status_ == closed)
          {
            break;
          }

          // if there are items, pop the next one
          if(!items_.empty())
          {
            // get the next item
            item = std::move(items_.front());
            items_.pop();

            notifier_.notify_one(status_, (int)(items_.empty() ? open_and_empty : open_and_ready));

            return true;
          }
        }
      }

      return false;
    }

  private:
    enum status
    {
      open_and_empty = 0,
      open_and_ready = 1,
      closed = 2
    };

    std::queue<T> items_;
    std::mutex mutex_;
    std::atomic<int> num_poppers_;

    // XXX synchronic<status> doesn't seem to work correctly
    //std::atomic<status> status_;
    //std::experimental::synchronic<status, std::experimental::synchronic_option::optimize_for_long_wait> notifier_;

    std::atomic<int> status_;
    std::experimental::synchronic<int, std::experimental::synchronic_option::optimize_for_short_wait> notifier_;
};


template<class T>
class condition_variable_concurrent_queue
{
  public:
    condition_variable_concurrent_queue()
      : is_closed_(false),
        num_poppers_(0)
    {
    }

    ~condition_variable_concurrent_queue()
    {
      close();
    }

    void close()
    {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        is_closed_ = true;
      }

      // wake everyone up
      wake_up_.notify_all();

      // wait until all the poppers have finished with wait_and_pop() 
      detail::wait_until_equal(num_poppers_, 0);
    }

    bool is_closed()
    {
      std::unique_lock<std::mutex> lock(mutex_);

      return is_closed_;
    }

    template<class... Args>
    queue_status emplace(Args&&... args)
    {
      {
        std::unique_lock<std::mutex> lock(mutex_);

        if(is_closed_)
        {
          return queue_status::closed;
        }

        items_.emplace(std::forward<Args>(args)...);
      }

      wake_up_.notify_one(); 

      return queue_status::open_and_ready;
    }

    queue_status push(const T& item)
    {
      return emplace(item);
    }

    // XXX this should return queue_status
    bool wait_and_pop(T& item)
    {
      scope_bumper<int> popping_(num_poppers_);

      while(true)
      {
        bool needs_notify = true;

        {
          std::unique_lock<std::mutex> lock(mutex_);
          wake_up_.wait(lock, [this]
          {
            return is_closed_ || !items_.empty();
          });

          // if the queue is closed, return
          if(is_closed_)
          {
            break;
          }

          // if there are no items go back to sleep
          if(items_.empty()) continue;

          // get the next item
          item = std::move(items_.front());
          items_.pop();

          needs_notify = !items_.empty();
        }

        // wake someone up
        if(needs_notify)
        {
          wake_up_.notify_one();
        }

        return true;
      }

      return false;
    }

  private:
    bool is_closed_;
    std::queue<T> items_;
    std::mutex mutex_;
    std::condition_variable wake_up_;
    std::atomic<int> num_poppers_;
};


template<class T>
using concurrent_queue = synchronic_concurrent_queue<T>;


} // end detail
} // end agency

