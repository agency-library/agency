#pragma once

#include <agency/cuda/detail/asynchronous_state.hpp>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class Function, class ResultPointer, class ArgumentPointer>
struct continuation
{
  mutable Function f_;
  ResultPointer result_ptr_;
  ArgumentPointer arg_ptr_;

  template<class Function1>
  __device__
  static void impl(Function1 f, unit, unit)
  {
    f();
  }

  template<class Function1, class T>
  __device__
  static void impl(Function1 f, T& result, unit)
  {
    result = f();
  }

  template<class Function1, class T>
  __device__
  static void impl(Function1 f, unit, T& arg)
  {
    f(arg);
  }

  template<class Function1, class T1, class T2>
  __device__
  static void impl(Function1 f, T1& result, T2& arg)
  {
    result = f(arg);
  }

  __device__
  void operator()() const
  {
    impl(f_, *result_ptr_, *arg_ptr_);
  }
};


template<class Function>
__host__ __device__
Function make_continuation(Function f, empty_type_ptr<void>, empty_type_ptr<void>)
{
  return f;
}

template<class Function, class ResultPointer, class ArgumentPointer>
__host__ __device__
continuation<Function,ResultPointer,ArgumentPointer> make_continuation(Function f, ResultPointer result_ptr, ArgumentPointer arg_ptr)
{
  return continuation<Function,ResultPointer,ArgumentPointer>{f, result_ptr, arg_ptr};
}


} // end detail
} // end cuda
} // end agency

