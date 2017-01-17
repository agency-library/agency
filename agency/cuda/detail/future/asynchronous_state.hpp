#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/asynchronous_state.hpp>
#include <agency/cuda/memory/detail/any_deleter.hpp>
#include <agency/cuda/memory/detail/unique_ptr.hpp>
#include <type_traits>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class T>
using asynchronous_state = agency::detail::asynchronous_state<T,cuda::detail::any_small_deleter<T>>;


// XXX it might be better for the following functions to be members of asynchronous_state
//     it also might better to have asynchronous_state's destructor asynchronously destroy the contained state
//     that way, we wouldn't have to make special arrangements inside of clients of asynchronous_state

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

