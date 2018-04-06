#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/execution_policy.hpp>
#include <agency/execution/executor/properties/bulk_guarantee.hpp>
#include <agency/future/always_ready_future.hpp>


namespace agency
{
namespace omp
{


class parallel_for_executor
{
  public:
    template<class T>
    using future = always_ready_future<T>;

    constexpr static bulk_guarantee_t::parallel_t query(const bulk_guarantee_t&)
    {
      return bulk_guarantee_t::parallel_t();
    }

    template<class Function, class ResultFactory, class SharedFactory>
    future<agency::detail::result_of_t<ResultFactory()>>
      bulk_twoway_execute(Function f, size_t n, ResultFactory result_factory, SharedFactory shared_factory) const
    {
#ifndef _OPENMP
      static_assert(sizeof(Function) && false, "agency::omp::parallel_for_executor requires C++ OpenMP language extensions (typically enabled with -fopenmp or /openmp).");
#endif

      auto result = result_factory();
      auto shared_parm = shared_factory();

      #pragma omp parallel for
      for(size_t i = 0; i < n; ++i)
      {
        f(i, result, shared_parm);
      }

      return agency::make_always_ready_future(std::move(result));
    }
};


using parallel_executor = parallel_for_executor;


class simd_executor
{
  public:
    template<class T>
    using future = always_ready_future<T>;

    template<class Function, class ResultFactory, class SharedFactory>
    future<agency::detail::result_of_t<ResultFactory()>>
      bulk_twoway_execute(Function f, size_t n, ResultFactory result_factory, SharedFactory shared_factory) const
    {
#if _OPENMP < 201307
      static_assert(sizeof(Function) && false, "agency::omp::simd_executor requires C++ OpenMP 4.0 or better language extensions (typically enabled with -fopenmp or /openmp).");
#endif

      auto result = result_factory();
      auto shared_parm = shared_factory();

      #pragma omp simd
      for(size_t i = 0; i < n; ++i)
      {
        f(i, result, shared_parm);
      }

      return agency::make_always_ready_future(std::move(result));
    }
};


using unsequenced_executor = simd_executor;


class parallel_execution_policy : public basic_execution_policy<parallel_agent, omp::parallel_executor, parallel_execution_policy>
{
  private:
    using super_t = basic_execution_policy<parallel_agent, omp::parallel_executor, parallel_execution_policy>;

  public:
    using super_t::basic_execution_policy;
};


const parallel_execution_policy par{};


class unsequenced_execution_policy : public basic_execution_policy<unsequenced_agent, omp::unsequenced_executor, unsequenced_execution_policy>
{
  private:
    using super_t = basic_execution_policy<unsequenced_agent, omp::unsequenced_executor, unsequenced_execution_policy>;

  public:
    using super_t::basic_execution_policy;
};


const unsequenced_execution_policy unseq{};


} // end omp
} // end agency

