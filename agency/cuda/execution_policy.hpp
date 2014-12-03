#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution_policy.hpp>
#include <agency/flattened_executor.hpp>
#include <agency/detail/ignore.hpp>
#include <type_traits>
#include <agency/cuda/execution_agent.hpp>
#include <agency/cuda/grid_executor.hpp>
#include <agency/cuda/block_executor.hpp>
#include <agency/cuda/parallel_executor.hpp>
#include <agency/cuda/detail/bind.hpp>


namespace agency
{
namespace cuda
{
namespace detail
{


template<class Function, class ExecutionAgentTraits, class ExecutorShape, class AgentShape>
struct execute_agent_functor
{
  __agency_hd_warning_disable__
  __host__ __device__
  execute_agent_functor(const Function& f,
                        const typename ExecutionAgentTraits::param_type& param,
                        const ExecutorShape& executor_shape,
                        const AgentShape& agent_shape)
    : f_(f),
      param_(param),
      executor_shape_(executor_shape),
      agent_shape_(agent_shape)
  {}

  template<class ExecutorIndex, class SharedParam>
  __device__
  void operator()(ExecutorIndex executor_idx, SharedParam&& shared_params) const
  {
    using agent_index_type = typename ExecutionAgentTraits::index_type;
    auto agent_idx = agency::detail::index_cast<agent_index_type>(executor_idx, executor_shape_, agent_shape_);

    ExecutionAgentTraits::execute(f_, agent_idx, param_, shared_params);
  }

  template<class ExecutorIndex, class SharedParam>
  __device__
  void operator()(ExecutorIndex executor_idx, SharedParam&& shared_params)
  {
    using agent_index_type = typename ExecutionAgentTraits::index_type;
    auto agent_idx = agency::detail::index_cast<agent_index_type>(executor_idx, executor_shape_, agent_shape_);

    ExecutionAgentTraits::execute(f_, agent_idx, param_, shared_params);
  }


  template<class ExecutorIndex>
  __device__
  void operator()(ExecutorIndex executor_idx, decltype(agency::detail::ignore)) const
  {
    using agent_index_type = typename ExecutionAgentTraits::index_type;
    auto agent_idx = agency::detail::index_cast<agent_index_type>(executor_idx, executor_shape_, agent_shape_);

    ExecutionAgentTraits::execute(f_, agent_idx, param_);
  }

  template<class ExecutorIndex>
  __device__
  void operator()(ExecutorIndex executor_idx, decltype(agency::detail::ignore))
  {
    using agent_index_type = typename ExecutionAgentTraits::index_type;
    auto agent_idx = agency::detail::index_cast<agent_index_type>(executor_idx, executor_shape_, agent_shape_);

    ExecutionAgentTraits::execute(f_, agent_idx, param_);
  }


  template<class ExecutorIndex>
  __device__
  void operator()(ExecutorIndex executor_idx) const
  {
    using agent_index_type = typename ExecutionAgentTraits::index_type;
    auto agent_idx = agency::detail::index_cast<agent_index_type>(executor_idx, executor_shape_, agent_shape_);

    ExecutionAgentTraits::execute(f_, agent_idx, param_);
  }

  template<class ExecutorIndex>
  __device__
  void operator()(ExecutorIndex executor_idx)
  {
    using agent_index_type = typename ExecutionAgentTraits::index_type;
    auto agent_idx = agency::detail::index_cast<agent_index_type>(executor_idx, executor_shape_, agent_shape_);

    ExecutionAgentTraits::execute(f_, agent_idx, param_);
  }

  Function f_;
  typename ExecutionAgentTraits::param_type param_;
  ExecutorShape executor_shape_;
  AgentShape agent_shape_;
};


template<class ExecutionAgentTraits, class Function, class ExecutorShape, class AgentShape>
__host__ __device__
execute_agent_functor<Function, ExecutionAgentTraits, ExecutorShape, AgentShape>
  make_execute_agent_functor(const Function& f,const typename ExecutionAgentTraits::param_type& param, 
                             const ExecutorShape& executor_shape, const AgentShape& agent_shape)
{
  return execute_agent_functor<Function,ExecutionAgentTraits,ExecutorShape,AgentShape>(f,param,executor_shape,agent_shape);
}


template<class ExecutionPolicy, class Function>
void bulk_invoke(const ExecutionPolicy& exec, Function&& f)
{
  using execution_agent_type = typename ExecutionPolicy::execution_agent_type;
  using traits = execution_agent_traits<execution_agent_type>;

  auto param = exec.param();
  auto agent_shape = traits::domain(param).shape();
  auto shared_init = traits::make_shared_initializer(param);

  using executor_type = typename ExecutionPolicy::executor_type;
  using executor_index_type = typename executor_traits<executor_type>::index_type;

  // convert the shape of the agent into the type of the executor's shape
  using executor_shape_type = typename executor_traits<executor_type>::shape_type;
  executor_shape_type executor_shape = agency::detail::shape_cast<executor_shape_type>(agent_shape);

  auto execute_me = make_execute_agent_functor<traits>(f, param, executor_shape, agent_shape);

  return executor_traits<executor_type>::bulk_invoke(exec.executor(), execute_me, executor_shape, shared_init);
}


// add basic_execution_policy to allow us to catch its derivations as overloads of bulk_invoke, etc.
template<class ExecutionAgent,
         class BulkExecutor,
         class ExecutionCategory = typename execution_agent_traits<ExecutionAgent>::execution_category,
         class DerivedExecutionPolicy = void>
class basic_execution_policy : public agency::detail::basic_execution_policy<ExecutionAgent,BulkExecutor,ExecutionCategory,DerivedExecutionPolicy>
{
  public:
    using agency::detail::basic_execution_policy<ExecutionAgent,BulkExecutor,ExecutionCategory,DerivedExecutionPolicy>::basic_execution_policy;
}; // end basic_execution_policy


} // end detail


class parallel_execution_policy : public detail::basic_execution_policy<cuda::parallel_agent, cuda::parallel_executor, parallel_execution_tag, parallel_execution_policy>
{
  public:
    using detail::basic_execution_policy<cuda::parallel_agent, cuda::parallel_executor, parallel_execution_tag, parallel_execution_policy>::basic_execution_policy;
};


const parallel_execution_policy par{};


class concurrent_execution_policy : public detail::basic_execution_policy<cuda::concurrent_agent, cuda::block_executor, concurrent_execution_tag, concurrent_execution_policy>
{
  public:
    using detail::basic_execution_policy<cuda::concurrent_agent, cuda::block_executor, concurrent_execution_tag, concurrent_execution_policy>::basic_execution_policy;
};


const concurrent_execution_policy con{};


template<class ExecutionPolicy, class Function, class... Args>
void bulk_invoke(const ExecutionPolicy& exec, Function&& f, Args&&... args)
{
  auto g = detail::bind(f, thrust::placeholders::_1, std::forward<Args>(args)...);
  detail::bulk_invoke(exec, g);
}


template<class ExecutionPolicy, class Function, class... Args>
std::future<void> bulk_async(const ExecutionPolicy& exec, Function&& f, Args&&... args)
{
  std::cout << "cuda::bulk_async(ExecutionPolicy): implement me!" << std::endl;
  return agency::detail::make_ready_future();
}


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


// specialize rebind_executor for cuda::block_executor
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


template<class ExecutionPolicy>
struct rebind_executor<ExecutionPolicy, cuda::grid_executor>
{
  using type = typename cuda::detail::rebind_executor_impl<ExecutionPolicy>::type;
};


template<class ExecutionPolicy>
struct rebind_executor<ExecutionPolicy, cuda::block_executor>
{
  using type = typename cuda::detail::rebind_executor_impl<ExecutionPolicy>::type;
};


// the following functions are overloads of agency::bulk_invoke & agency::bulk_async
// for cuda execution policies. they forward along the to cuda::bulk_invoke & cuda::bulk_async
// we introduce these overloads to work around the use of lambdas in agency::bulk_invoke & agency::bulk_async

template<class ExecutionAgent, class BulkExecutor, class ExecutionCategory, class DerivedExecutionPolicy, class Function, class... Args>
void bulk_invoke(cuda::detail::basic_execution_policy<ExecutionAgent,BulkExecutor,ExecutionCategory,DerivedExecutionPolicy>&& exec,
                 Function&& f,
                 Args&&... args)
{
  return cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class ExecutionAgent, class BulkExecutor, class ExecutionCategory, class DerivedExecutionPolicy, class Function, class... Args>
void bulk_invoke(cuda::detail::basic_execution_policy<ExecutionAgent,BulkExecutor,ExecutionCategory,DerivedExecutionPolicy>& exec,
                 Function&& f,
                 Args&&... args)
{
  return cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class ExecutionAgent, class BulkExecutor, class ExecutionCategory, class DerivedExecutionPolicy, class Function, class... Args>
void bulk_invoke(const cuda::detail::basic_execution_policy<ExecutionAgent,BulkExecutor,ExecutionCategory,DerivedExecutionPolicy>& exec,
                 Function&& f,
                 Args&&... args)
{
  return cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}


template<class ExecutionAgent, class BulkExecutor, class ExecutionCategory, class DerivedExecutionPolicy, class Function, class... Args>
std::future<void> bulk_async(cuda::detail::basic_execution_policy<ExecutionAgent,BulkExecutor,ExecutionCategory,DerivedExecutionPolicy>&& exec, Function&& f, Args&&... args)
{
  return cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class ExecutionAgent, class BulkExecutor, class ExecutionCategory, class DerivedExecutionPolicy, class Function, class... Args>
std::future<void> bulk_async(cuda::detail::basic_execution_policy<ExecutionAgent,BulkExecutor,ExecutionCategory,DerivedExecutionPolicy>& exec, Function&& f, Args&&... args)
{
  return cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class ExecutionAgent, class BulkExecutor, class ExecutionCategory, class DerivedExecutionPolicy, class Function, class... Args>
std::future<void> bulk_async(const cuda::detail::basic_execution_policy<ExecutionAgent,BulkExecutor,ExecutionCategory,DerivedExecutionPolicy>& exec, Function&& f, Args&&... args)
{
  return cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}


} // end agency

