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


template<class T>
class synchronic_concurrent_queue
{
  public:
    synchronic_concurrent_queue()
      : status_(active_and_empty)
    {
    }

    ~synchronic_concurrent_queue()
    {
      close();
    }

    void close()
    {
      std::unique_lock<std::mutex> lock(mutex_);

      // notify that we're closing
      notifier_.notify_all(status_, (int)inactive);
    }

    template<class... Args>
    void emplace(Args&&... args)
    {
      std::unique_lock<std::mutex> lock(mutex_);

      items_.emplace(std::forward<Args>(args)...);

      notifier_.notify_one(status_, (int)active_and_ready); 
    }

    void push(const T& item)
    {
      emplace(item);
    }

    bool wait_and_pop(T& item)
    {
      while(true)
      {
        notifier_.wait_for_change(status_, (int)active_and_empty);

        {
          std::unique_lock<std::mutex> lock(mutex_);

          // if the queue is closed, return
          if(status_ == inactive)
          {
            return false;
          }

          // if there are no items go back to sleep
          if(items_.empty())
          {
            continue;
          }

          // get the next item
          item = std::move(items_.front());
          items_.pop();

          notifier_.notify_one(status_, (int)(items_.empty() ? active_and_empty : active_and_ready));
        }

        return true;
      }

      return false;
    }

  private:
    enum status
    {
      active_and_empty = 0,
      active_and_ready = 1,
      inactive = 2
    };

    std::queue<T> items_;
    std::mutex mutex_;

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
      : is_closed_(false)
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
    }

    template<class... Args>
    void emplace(Args&&... args)
    {
      {
        std::unique_lock<std::mutex> lock(mutex_);

        items_.emplace(std::forward<Args>(args)...);
      }

      wake_up_.notify_one(); 
    }

    void push(const T& item)
    {
      emplace(item);
    }

    bool wait_and_pop(T& item)
    {
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
          if(is_closed_) return false;

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
};


template<class T>
using concurrent_queue = synchronic_concurrent_queue<T>;


} // end detail
} // end agency

