#pragma once

#include <future>
#include <agency/execution_categories.hpp>
#include <agency/future.hpp>
#include <utility>


namespace agency
{
namespace this_thread
{


class vector_executor
{
  public:
    using execution_category = vector_execution_tag;

    template<class Function, class Factory>
    void execute(Function f, size_t n, Factory shared_factory)
    {
      auto shared_parm = shared_factory();

      // ivdep requires gcc 4.9+
#if !defined(__INTEL_COMPILER) && !defined(__NVCC__) && (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 9)
      #pragma GCC ivdep
#elif defined(__INTEL_COMPILER)
      #pragma simd
#endif
      for(size_t i = 0; i < n; ++i)
      {
        f(i, shared_parm);
      }
    }

    template<class Function, class Factory>
    std::future<void> async_execute(Function f, size_t n, Factory shared_factory)
    {
      return std::async(std::launch::deferred, [=]
      {
        this->execute(f, n, shared_factory);
      });
    }

    template<class Function, class Factory>
    std::future<void> then_execute(Function f, size_t n, std::future<void>& fut, Factory shared_factory)
    {
      return detail::then(fut, std::launch::deferred, [=](std::future<void>& predecessor)
      {
        this->execute(f, n, shared_factory);
      });
    }

    template<class T, class Function, class Factory>
    std::future<void> then_execute(Function f, size_t n, std::future<void>& fut, Factory shared_factory)
    {
      return detail::then(fut, std::launch::deferred, [=](std::future<T>& predecessor) mutable
      {
        using second_type = decltype(shared_factory);

        auto first = predecessor.get();

        this->execute([=](size_t idx, std::pair<T,second_type>& p) mutable
        {
          f(idx, p.first, p.second);
        },
        n,
        [=]
        {
          return std::make_pair(first, shared_factory());
        });
      });
    }
};


} // end this_thread


// XXX consider a flattened nesting similar to parallel_executor
using vector_executor = this_thread::vector_executor;


} // end agency

