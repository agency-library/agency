#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/asynchronous_state.hpp>
#include <agency/cuda/memory/detail/unique_ptr.hpp>
#include <agency/cuda/memory/allocator/detail/any_allocator.hpp>
#include <type_traits>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class T>
using asynchronous_state = agency::detail::asynchronous_state<T,cuda::detail::any_small_allocator<T>>;


// XXX consider giving this function an overload taking an allocator
__agency_exec_check_disable__
template<class Factory>
__host__ __device__
asynchronous_state<agency::detail::result_of_t<Factory()>>
  make_asynchronous_state(Factory factory)
{
  using type = agency::detail::result_of_t<Factory()>;

  // XXX calling factory() here requires that its result be moveable
  //     however, types such as synchronization primitives aren't moveable
  //     it might be a better idea to push this call to the factory() down into
  //     asynchronous_state's constructor so that it can initialize the result of factory() directly
  return asynchronous_state<type>(agency::detail::construct_ready, agency::cuda::allocator<type>(), factory());
}


// XXX it might be better for the following functions to be members of asynchronous_state
//     it also might better to have asynchronous_state's destructor asynchronously destroy the contained state
//     that way, we wouldn't have to make special arrangements inside of clients of asynchronous_state

// XXX shouldn't the following functions also take an Allocator template parameter for asynchronous_state?

template<class T>
__host__ __device__
typename std::enable_if<
  std::is_void<typename asynchronous_state<T>::storage_type>::value
>::type
  invalidate_and_destroy_when(asynchronous_state<T>& state, event&)
{
  // assign an default-constructed state to invalidate it
  state = asynchronous_state<T>();

  // destruction is a no-op
}


template<class T>
__host__ __device__
typename std::enable_if<
  !std::is_void<typename asynchronous_state<T>::storage_type>::value
>::type
  invalidate_and_destroy_when(asynchronous_state<T>& state, event& e)
{
  // releasing the state's storage implicitly invalidates it
  agency::cuda::detail::release_and_delete_when(state.storage(), e);
}


template<class T>
__host__ __device__
void async_invalidate_and_destroy(asynchronous_state<T>& state)
{
  auto immediately_ready = detail::make_ready_event();

  agency::cuda::detail::invalidate_and_destroy_when(state, immediately_ready);
}

  
} // end detail
} // end cuda
} // end agency

