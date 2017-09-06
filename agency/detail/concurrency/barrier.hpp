#pragma once

#include <agency/detail/config.hpp>

#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

namespace agency
{
namespace detail
{


class blocking_barrier
{
  public:
    inline explicit blocking_barrier(size_t num_threads)
      : unarrived_count_(num_threads),
        count_(num_threads),
        generation_(0)

    {
      if(num_threads == 0) throw std::invalid_argument("barrier: num_threads may not be 0.");
    }

    // define this to workaround nvcc's automatic execution space deduction for compiler-generated functions
    inline ~blocking_barrier() {}

    inline size_t count() const
    {
      return count_;
    }

    inline void arrive_and_drop()
    {
      std::unique_lock<std::mutex> lock(mutex_);

      if(--unarrived_count_ == 0)
      {
        // initialize the unarrived count variable
        unarrived_count_ = count();

        // bump the generation number
        generation_++;

        // unblock all blocking threads
        cv_.notify_all();
      }
    }

    inline void arrive_and_wait()
    {
      std::unique_lock<std::mutex> lock(mutex_);

      if(--unarrived_count_ == 0)
      {
        // initialize the unarrived count variable
        unarrived_count_ = count();

        // bump the generation number
        generation_++;

        // unblock all blocking threads
        cv_.notify_all();
      }
      else
      {
        // block until either we are woken or the generation changes
        size_t old_generation = generation_;
        cv_.wait(lock, [=]{ return this->generation_ != old_generation; });
      }
    }

  private:
    size_t                  unarrived_count_;
    size_t                  count_;
    size_t                  generation_;
    std::mutex              mutex_;
    std::condition_variable cv_;
};


class spinning_barrier
{
  public:
    inline explicit spinning_barrier(size_t num_threads)
      : count_(num_threads), 
        num_spinning_(0),
        generation_(0)
    {
      if(num_threads == 0) throw std::invalid_argument("barrier: num_threads may not be 0.");
    }

    // define this to workaround nvcc's automatic execution space deduction for compiler-generated functions
    inline ~spinning_barrier() {}

    inline size_t count() const
    {
      return count_;
    }

    inline void arrive_and_drop()
    {
      if(num_spinning_.fetch_add(1) == count_ - 1)
      {
        num_spinning_.store(0);
        generation_.fetch_add(1);
      }
    }

    inline void arrive_and_wait()
    {
      size_t generation = generation_.load();

      if(num_spinning_.fetch_add(1) == count_ - 1)
      {
        num_spinning_.store(0);
        generation_.fetch_add(1);
      }
      else
      {
        while(generation_.load() == generation)
        {
          ;
        }
      }
    }

protected:
    size_t              count_;
    std::atomic<size_t> num_spinning_;
    std::atomic<size_t> generation_;
};


using barrier = blocking_barrier;


} // end detail
} // end agency

