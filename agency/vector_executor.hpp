#pragma once

#include <future>
#include <agency/execution_categories.hpp>
#include <agency/functional.hpp>
#include <functional>


namespace agency
{
namespace this_thread
{


class vector_executor
{
  public:
    using execution_category = vector_execution_tag;

    template<class Function, class T>
    void execute(Function f, size_t n, T&& shared_init)
    {
      auto shared_parm = agency::decay_construct(std::forward<T>(shared_init));

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

    template<class Function, class T>
    std::future<void> async_execute(Function f, size_t n, T&& shared_init)
    {
      return std::async(std::launch::deferred, [=]
      {
        this->execute(f, n, shared_init);
      });
    }

    template<class Function, class T>
    std::future<void> then_execute(std::future<void>& fut, Function f, size_t n, T&& shared_init)
    {
      return detail::then(fut, std::launch::deferred, [=](std::future<void>& predecessor)
      {
        this->execute(f, n, shared_init);
      });
    }

    template<class T1, class Function, class T2>
    std::future<void> then_execute(std::future<T1>& fut, Function f, size_t n, T2&& shared_init)
    {
      return detail::then(fut, std::launch::deferred, [=](std::future<T1>& predecessor) mutable
      {
        using second_type = typename std::decay<T2>::type;

        this->execute([=](size_t idx, std::pair<T1,second_type>& p) mutable
        {
          f(idx, p.first, p.second);
        },
        n,
        std::make_pair(std::move(predecessor.get()), std::move(shared_init))
        );
      });
    }
};


} // end this_thread


// XXX consider a flattened nesting similar to parallel_executor
using vector_executor = this_thread::vector_executor;


} // end agency

