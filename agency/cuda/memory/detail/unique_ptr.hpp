#pragma once

#include <agency/detail/config.hpp>
#include <agency/memory/detail/allocation_deleter.hpp>
#include <agency/memory/detail/unique_ptr.hpp>
#include <agency/cuda/memory/allocator/allocator.hpp>
#include <agency/cuda/detail/future/event.hpp>
#include <agency/cuda/detail/workaround_unused_variable_warning.hpp>
#include <agency/async.hpp>
#include <mutex>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class T>
class default_delete : public agency::detail::allocation_deleter<cuda::allocator<T>>
{
  private:
    using super_t = agency::detail::allocation_deleter<cuda::allocator<T>>;

  public:
    using super_t::super_t;

    __AGENCY_ANNOTATION
    default_delete()
      : super_t(cuda::allocator<T>())
    {}
};


template<class T, class Deleter = default_delete<T>>
using unique_ptr = agency::detail::unique_ptr<T, Deleter>;


template<class T, class... Args>
__host__ __device__
unique_ptr<T> make_unique(Args&&... args)
{
  return agency::detail::allocate_unique<T>(allocator<T>(),std::forward<Args>(args)...);
}


template<class T, class Deleter>
struct release_and_delete_when_functor
{
  T* ptr;
  mutable Deleter deleter;

  struct task
  {
    T* ptr;
    mutable Deleter deleter;

    void operator()() const
    {
      deleter(ptr);
    }
  };

  __host__ __device__
  void operator()() const
  {
#ifndef __CUDA_ARCH__
    // this functor is called in a stream callback thread where it's not permitted to make
    // calls back into CUDART
    // execute the deletion task on a different thread
    // XXX agency::async() isn't actually guaranteed to execute the task in a separate thread
    agency::async(task{ptr,deleter});
#else
    deleter(ptr);
#endif
  }
};


template<class T, class Deleter>
__host__ __device__
void release_and_delete_when(unique_ptr<T,Deleter>& ptr, event& e)
{
  auto continuation = release_and_delete_when_functor<T,Deleter>{ptr.release(), ptr.get_deleter()};

  auto delete_event = e.then(continuation);

  if(!delete_event.valid())
  {
    printf("release_and_delete_when(): invalid delete_event\n");
  }

  assert(delete_event.valid());

#ifndef __CUDA_ARCH__
  // collect all events created and ensure they complete before the program exits
  // to ensure that T's destructor is called
  static std::vector<blocking_event> all_events;
  static std::mutex mutex;

  std::unique_lock<std::mutex> guard(mutex);
  all_events.emplace_back(std::move(delete_event));
#else
  detail::workaround_unused_variable_warning(delete_event);
#endif
}


} // end detail
} // end cuda
} // end agency

