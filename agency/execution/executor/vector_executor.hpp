#pragma once

#include <agency/detail/config.hpp>
#include <agency/future/always_ready_future.hpp>
#include <utility>


namespace agency
{
namespace this_thread
{


class vector_executor
{
  public:
    template<class T>
    using future = always_ready_future<T>;

    template<class Function, class ResultFactory, class SharedFactory>
    future<agency::detail::result_of_t<ResultFactory()>>
      bulk_twoway_execute(Function f, size_t n, ResultFactory result_factory, SharedFactory shared_factory) const
    {
      auto result = result_factory();
      auto shared_parm = shared_factory();

      // ivdep requires gcc 4.9+
#if !defined(__INTEL_COMPILER) && !defined(__NVCC__) && (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 9)
      #pragma GCC ivdep
#elif defined(__INTEL_COMPILER)
      #pragma simd
#endif
      for(size_t i = 0; i < n; ++i)
      {
        f(i, result, shared_parm);
      }

      return agency::make_always_ready_future(std::move(result));
    }
};


} // end this_thread


// XXX consider a flattened nesting similar to parallel_executor
using vector_executor = this_thread::vector_executor;


} // end agency

