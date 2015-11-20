#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution_policy.hpp>
#include <agency/flattened_executor.hpp>
#include <agency/cuda/execution_agent.hpp>
#include <agency/cuda/grid_executor.hpp>
#include <agency/cuda/parallel_executor.hpp>
#include <agency/cuda/concurrent_executor.hpp>
#include <agency/cuda/nested_executor.hpp>
#include <agency/cuda/bulk_invoke.hpp>
#include <agency/detail/tuple.hpp>
#include <type_traits>


namespace agency
{
namespace cuda
{
namespace detail
{




// add basic_execution_policy to allow us to catch its derivations as overloads of bulk_invoke, etc.
template<class ExecutionAgent,
         class BulkExecutor,
         class ExecutionCategory = typename execution_agent_traits<ExecutionAgent>::execution_category,
         class DerivedExecutionPolicy = void>
class basic_execution_policy :
  public agency::detail::basic_execution_policy<
    ExecutionAgent,
    BulkExecutor,
    ExecutionCategory,
    typename std::conditional<
      std::is_void<DerivedExecutionPolicy>::value,
      basic_execution_policy<ExecutionAgent,BulkExecutor,ExecutionCategory,DerivedExecutionPolicy>,
      DerivedExecutionPolicy
    >::type
>
{
  public:
    // inherit the base class's constructors
    using agency::detail::basic_execution_policy<
      ExecutionAgent,
      BulkExecutor,
      ExecutionCategory,
      typename std::conditional<
        std::is_void<DerivedExecutionPolicy>::value,
        basic_execution_policy<ExecutionAgent,BulkExecutor,ExecutionCategory,DerivedExecutionPolicy>,
        DerivedExecutionPolicy
      >::type
    >::basic_execution_policy;
}; // end basic_execution_policy


} // end detail




class parallel_execution_policy : public detail::basic_execution_policy<cuda::parallel_agent, cuda::parallel_executor, parallel_execution_tag, parallel_execution_policy>
{
  private:
    using super_t = detail::basic_execution_policy<cuda::parallel_agent, cuda::parallel_executor, parallel_execution_tag, parallel_execution_policy>;

    using par2d_t = detail::basic_execution_policy<cuda::parallel_agent_2d, cuda::parallel_executor, parallel_execution_tag>;

  public:
    using super_t::basic_execution_policy;

    using super_t::operator();

    // XXX consider whether we really want this functionality a member of parallel_execution_policy
    inline par2d_t operator()(agency::size2 shape) const
    {
      par2d_t par2d;

      return par2d(shape);
    }

    // XXX consider whether we really want this functionality a member of parallel_execution_policy
    template<class ExecutionPolicy>
    agency::detail::nested_execution_policy<
      par2d_t,
      agency::detail::decay_t<ExecutionPolicy>
    >
      operator()(agency::size2 domain, ExecutionPolicy&& exec) const
    {
      par2d_t par2d;

      return par2d(domain, std::forward<ExecutionPolicy>(exec));
    }
};


const parallel_execution_policy par{};


class concurrent_execution_policy : public detail::basic_execution_policy<cuda::concurrent_agent, cuda::concurrent_executor, concurrent_execution_tag, concurrent_execution_policy>
{
  private:
    using super_t = detail::basic_execution_policy<cuda::concurrent_agent, cuda::concurrent_executor, concurrent_execution_tag, concurrent_execution_policy>;

    using con2d_t = detail::basic_execution_policy<cuda::concurrent_agent_2d, cuda::concurrent_executor, concurrent_execution_tag>;

  public:
    using super_t::basic_execution_policy;

    using super_t::operator();

    // XXX consider whether we really want this functionality a member of concurrent_execution_policy
    inline con2d_t operator()(agency::size2 shape) const
    {
      con2d_t con2d;

      return con2d(shape);
    }

    // XXX consider whether we really want this functionality a member of concurrent_execution_policy
    template<class ExecutionPolicy>
    agency::detail::nested_execution_policy<
      con2d_t,
      agency::detail::decay_t<ExecutionPolicy>
    >
      operator()(agency::size2 domain, ExecutionPolicy&& exec) const
    {
      con2d_t con2d;

      return con2d(domain, std::forward<ExecutionPolicy>(exec));
    }
};


const concurrent_execution_policy con{};


namespace detail
{


// specialize rebind_executor for cuda's executor types
template<class ExecutionPolicy, class Enable = void>
struct rebind_executor_impl;


template<class ExecutionPolicy>
struct rebind_executor_impl<
  ExecutionPolicy,
  typename std::enable_if<
    std::is_same<
      typename ExecutionPolicy::execution_category,
      parallel_execution_tag
    >::value
  >::type
>
{
  using type = cuda::parallel_execution_policy;
};


// specialize rebind_executor for cuda::concurrent_executor
template<class ExecutionPolicy>
struct rebind_executor_impl<
  ExecutionPolicy,
  typename std::enable_if<
    std::is_same<
      typename ExecutionPolicy::execution_category,
      concurrent_execution_tag
    >::value
  >::type
>
{
  using type = cuda::concurrent_execution_policy;
};


} // end detail
} // end cuda


// specialize is_execution_policy
template<class ExecutionAgent,
         class BulkExecutor,
         class ExecutionCategory,
         class DerivedExecutionPolicy>
struct is_execution_policy<cuda::detail::basic_execution_policy<ExecutionAgent,BulkExecutor,ExecutionCategory,DerivedExecutionPolicy>> : std::true_type {};


template<>
struct is_execution_policy<cuda::parallel_execution_policy> : std::true_type {};


template<>
struct is_execution_policy<cuda::concurrent_execution_policy> : std::true_type {};


template<class ExecutionPolicy>
struct rebind_executor<ExecutionPolicy, cuda::grid_executor>
{
  using type = typename cuda::detail::rebind_executor_impl<ExecutionPolicy>::type;
};


template<class ExecutionPolicy>
struct rebind_executor<ExecutionPolicy, cuda::concurrent_executor>
{
  using type = typename cuda::detail::rebind_executor_impl<ExecutionPolicy>::type;
};


// the following functions are overloads of agency::bulk_invoke & agency::bulk_async
// for cuda execution policies. they forward along the to cuda::bulk_invoke & cuda::bulk_async
// we introduce these overloads to work around the use of lambdas in agency::bulk_invoke & agency::bulk_async
// which are obstacles to __global__ function template instantiation for nvcc

template<class ExecutionAgent, class BulkExecutor, class ExecutionCategory, class DerivedExecutionPolicy, class Function, class... Args>
auto bulk_invoke(cuda::detail::basic_execution_policy<ExecutionAgent,BulkExecutor,ExecutionCategory,DerivedExecutionPolicy>&& exec,
                 Function&& f,
                 Args&&... args)
  -> decltype(
       cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...)
     )
{
  return cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class ExecutionAgent, class BulkExecutor, class ExecutionCategory, class DerivedExecutionPolicy, class Function, class... Args>
auto bulk_invoke(cuda::detail::basic_execution_policy<ExecutionAgent,BulkExecutor,ExecutionCategory,DerivedExecutionPolicy>& exec,
                 Function&& f,
                 Args&&... args)
  -> decltype(
       cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...)
     )
{
  return cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class ExecutionAgent, class BulkExecutor, class ExecutionCategory, class DerivedExecutionPolicy, class Function, class... Args>
auto bulk_invoke(const cuda::detail::basic_execution_policy<ExecutionAgent,BulkExecutor,ExecutionCategory,DerivedExecutionPolicy>& exec,
                 Function&& f,
                 Args&&... args)
  -> decltype(
       cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...)
     )
{
  return cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}


template<class Function, class... Args>
auto bulk_invoke(cuda::parallel_execution_policy&& exec,
                 Function&& f,
                 Args&&... args)
  -> decltype(
       cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...)
     )
{
  return cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class Function, class... Args>
auto bulk_invoke(cuda::parallel_execution_policy& exec,
                 Function&& f,
                 Args&&... args)
  -> decltype(
       cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...)
     )
{
  return cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class Function, class... Args>
auto bulk_invoke(const cuda::parallel_execution_policy& exec,
                 Function&& f,
                 Args&&... args)
  -> decltype(
       cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...)
     )
{
  return cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}


template<class Function, class... Args>
auto bulk_invoke(cuda::concurrent_execution_policy&& exec,
                 Function&& f,
                 Args&&... args)
  -> decltype(
       cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...)
     )
{
  return cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class Function, class... Args>
auto bulk_invoke(cuda::concurrent_execution_policy& exec,
                 Function&& f,
                 Args&&... args)
  -> decltype(
       cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...)
     )
{
  return cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class Function, class... Args>
auto bulk_invoke(const cuda::concurrent_execution_policy& exec,
                 Function&& f,
                 Args&&... args)
  -> decltype(
       cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...)
     )
{
  return cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}


template<class ExecutionAgent, class BulkExecutor, class ExecutionCategory, class DerivedExecutionPolicy, class Function, class... Args>
auto bulk_async(cuda::detail::basic_execution_policy<ExecutionAgent,BulkExecutor,ExecutionCategory,DerivedExecutionPolicy>&& exec, Function&& f, Args&&... args)
  -> decltype(
       cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...)
     )
{
  return cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class ExecutionAgent, class BulkExecutor, class ExecutionCategory, class DerivedExecutionPolicy, class Function, class... Args>
auto bulk_async(cuda::detail::basic_execution_policy<ExecutionAgent,BulkExecutor,ExecutionCategory,DerivedExecutionPolicy>& exec, Function&& f, Args&&... args)
  -> decltype(
       cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...)
     )
{
  return cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class ExecutionAgent, class BulkExecutor, class ExecutionCategory, class DerivedExecutionPolicy, class Function, class... Args>
auto bulk_async(const cuda::detail::basic_execution_policy<ExecutionAgent,BulkExecutor,ExecutionCategory,DerivedExecutionPolicy>& exec, Function&& f, Args&&... args)
  -> decltype(
       cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...)
     )
{
  return cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}


template<class Function, class... Args>
auto bulk_async(cuda::parallel_execution_policy&& exec, Function&& f, Args&&... args)
  -> decltype(
       cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...)
     )
{
  return cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class Function, class... Args>
auto bulk_async(cuda::parallel_execution_policy& exec, Function&& f, Args&&... args)
  -> decltype(
       cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...)
     )
{
  return cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class Function, class... Args>
auto bulk_async(const cuda::parallel_execution_policy& exec, Function&& f, Args&&... args)
  -> decltype(
       cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...)
     )
{
  return cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}


template<class Function, class... Args>
auto bulk_async(cuda::concurrent_execution_policy&& exec, Function&& f, Args&&... args)
  -> decltype(
       cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...)
     )
{
  return cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class Function, class... Args>
auto bulk_async(cuda::concurrent_execution_policy& exec, Function&& f, Args&&... args)
  -> decltype(
       cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...)
     )
{
  return cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class Function, class... Args>
auto bulk_async(const cuda::concurrent_execution_policy& exec, Function&& f, Args&&... args)
  -> decltype(
       cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...)
     )
{
  return cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}


} // end agency

