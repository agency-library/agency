#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/optional.hpp>

#include <queue>
#include <atomic>
#include <condition_variable>

// synchronic requires c++14
#if __cplusplus > 201103L
#include <agency/detail/concurrency/synchronic>
#endif


namespace agency
{
namespace detail
{


#if __cplusplus > 201103L

template<class T>
class synchronic_concurrent_queue
{
  public:
    synchronic_concurrent_queue()
      : shutdown_(false),
        wake_up_(false)
    {
    }

    ~synchronic_concurrent_queue()
    {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        shutdown_ = true;
      }

      // wake everyone up
      notifier_.notify_all(wake_up_, true);
    }

    template<class... Args>
    void emplace(Args&&... args)
    {
      {
        std::unique_lock<std::mutex> lock(mutex_);

        items_.emplace(std::forward<Args>(args)...);
      }

      notifier_.notify_one(wake_up_, true); 
    }

    void push(const T& item)
    {
      emplace(item);
    }

    // XXX suspicious of the performance of optional<T> here
    agency::detail::optional<T> blocking_pop()
    {
      while(true)
      {
        notifier_.wait(wake_up_, true);

        agency::detail::optional<T> result = agency::detail::nullopt;
        bool needs_notify = false;

        {
          std::unique_lock<std::mutex> lock(mutex_);

          // if it's time to shutdown, return
          if(shutdown_ && items_.empty()) return agency::detail::nullopt;

          // if there are no items go back to sleep
          if(items_.empty()) continue;

          // get the next item
          result = std::move(items_.front());
          items_.pop();

          bool needs_notify = !items_.empty();
        }

        if(needs_notify)
        {
          notifier_.notify_one(wake_up_, true);
        }

        return std::move(result);
      }

      return agency::detail::nullopt;
    }

    bool blocking_pop(T& item)
    {
      while(true)
      {
        notifier_.wait(wake_up_, true);

        bool needs_notify = false;

        {
          std::unique_lock<std::mutex> lock(mutex_);

          // if it's time to shutdown, return
          if(shutdown_ && items_.empty()) return false;

          // if there are no items go back to sleep
          if(items_.empty()) continue;

          // get the next item
          item = std::move(items_.front());
          items_.pop();

          bool needs_notify = !items_.empty();
        }

        if(needs_notify)
        {
          notifier_.notify_one(wake_up_, true);
        }

        return true;
      }

      return false;
    }

  private:
    bool shutdown_;
    std::queue<T> items_;
    std::mutex mutex_;
    std::atomic<bool> wake_up_;
    std::experimental::synchronic<bool, std::experimental::synchronic_option::optimize_for_short_wait> notifier_;
};


#endif // c++14


template<class T>
class condition_variable_concurrent_queue
{
  public:
    condition_variable_concurrent_queue()
      : shutdown_(false)
    {
    }

    ~condition_variable_concurrent_queue()
    {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        shutdown_ = true;
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

    // XXX suspicious of the performance of optional<T> here
    agency::detail::optional<T> blocking_pop()
    {
      while(true)
      {
        agency::detail::optional<T> result = agency::detail::nullopt;
        bool needs_notify = true;

        {
          std::unique_lock<std::mutex> lock(mutex_);
          wake_up_.wait(lock, [this]
          {
            return shutdown_ || !items_.empty();
          });

          // if it's time to shutdown, return
          if(shutdown_ && items_.empty()) return agency::detail::nullopt;

          // if there are no items go back to sleep
          if(items_.empty()) continue;

          // get the next item
          result = std::move(items_.front());
          items_.pop();

          needs_notify = !items_.empty();
        }

        // wake someone up
        if(needs_notify)
          wake_up_.notify_one();

        return std::move(result);
      }

      return agency::detail::nullopt;
    }

    bool blocking_pop(T& item)
    {
      while(true)
      {
        bool needs_notify = true;

        {
          std::unique_lock<std::mutex> lock(mutex_);
          wake_up_.wait(lock, [this]
          {
            return shutdown_ || !items_.empty();
          });

          // if it's time to shutdown, return
          if(shutdown_ && items_.empty()) return false;

          // if there are no items go back to sleep
          if(items_.empty()) continue;

          // get the next item
          item = std::move(items_.front());
          items_.pop();
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
    bool shutdown_;
    std::queue<T> items_;
    std::mutex mutex_;
    std::condition_variable wake_up_;
};

#if __cplusplus > 201103L

template<class T>
using concurrent_queue = synchronic_concurrent_queue<T>;

#else

template<class T>
using concurrent_queue = condition_variable_concurrent_queue<T>;

#endif

} // end detail
} // end agency

