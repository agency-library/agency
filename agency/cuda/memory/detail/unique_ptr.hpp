#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/memory/unique_ptr.hpp>
#include <agency/cuda/memory/allocator.hpp>
#include <agency/cuda/detail/future/event.hpp>
#include <agency/cuda/detail/workaround_unused_variable_warning.hpp>
#include <mutex>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class T>
using unique_ptr = agency::detail::unique_ptr<T, agency::detail::deleter<allocator<T>>>;


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

  __host__ __device__
  void operator()() const
  {
    deleter(ptr);
  }
};


template<class T>
__host__ __device__
void release_and_delete_when(unique_ptr<T>& ptr, event& e)
{
  auto continuation = release_and_delete_when_functor<T,typename unique_ptr<T>::deleter_type>{ptr.release(), ptr.get_deleter()};

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

