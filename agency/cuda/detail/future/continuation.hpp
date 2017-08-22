#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/future/asynchronous_state.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/tuple.hpp>

namespace agency
{
namespace cuda
{
namespace detail
{


// XXX we may wish to do all of this inside of detail::cuda_kernel to avoid additional nested templates


template<class Function, class ResultPointer, class PointerTuple>
struct continuation
{
  mutable Function f_;
  mutable ResultPointer result_ptr_;
  mutable PointerTuple arg_ptr_tuple_;

  template<class Function1>
  __host__ __device__
  continuation(Function1&& f, ResultPointer result_ptr, PointerTuple arg_ptr_tuple)
    : f_(std::forward<Function1>(f)), result_ptr_(result_ptr), arg_ptr_tuple_(arg_ptr_tuple)
  {}

  template<size_t... Indices>
  __device__
  void impl(agency::detail::index_sequence<Indices...>) const
  {
    *result_ptr_ = f_(*agency::get<Indices>(arg_ptr_tuple_)...);
  }

  __device__
  void operator()() const
  {
    impl(agency::detail::make_index_sequence<std::tuple_size<PointerTuple>::value>{});
  }
};


template<class Function, class PointerTuple>
struct continuation<Function, agency::detail::empty_type_ptr<void>, PointerTuple>
{
  mutable Function f_;
  mutable PointerTuple arg_ptr_tuple_;

  template<class Function1>
  __host__ __device__
  continuation(Function1&& f, agency::detail::empty_type_ptr<void>, PointerTuple arg_ptr_tuple)
    : f_(std::forward<Function1>(f)), arg_ptr_tuple_(arg_ptr_tuple)
  {}

  template<size_t... Indices>
  __device__
  void impl(agency::detail::index_sequence<Indices...>) const
  {
    f_(*agency::get<Indices>(arg_ptr_tuple_)...);
  }

  __device__
  void operator()() const
  {
    impl(agency::detail::make_index_sequence<std::tuple_size<PointerTuple>::value>{});
  }
};


template<class Function, class ResultPointer, class PointerTuple>
__host__ __device__
continuation<typename std::decay<Function>::type,ResultPointer,PointerTuple>
  make_continuation(Function&& f, ResultPointer result_ptr, PointerTuple arg_ptr_tuple)
{
  return continuation<typename std::decay<Function>::type,ResultPointer,PointerTuple>(std::forward<Function>(f), result_ptr, arg_ptr_tuple);
}


} // end detail
} // end cuda
} // end agency

