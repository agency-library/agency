#pragma once

#include <execution_policy>
#include <flattened_executor>
#include <type_traits>
#include "grid_executor.hpp"
#include "bind.hpp"

namespace std
{
namespace __cuda
{


template<class T>
struct has_cuda_grid_executor
  : std::is_same<
      cuda::grid_executor,
      typename T::executor_type
    >
{
};


template<class ExecutionPolicy>
struct is_cuda_execution_policy
  : std::integral_constant<
      bool,
      std::is_execution_policy<ExecutionPolicy>::value &&
      has_cuda_grid_executor<ExecutionPolicy>::value
    >
{};


using execution_policy = __basic_execution_policy<
  parallel_agent,
  flattened_executor<cuda::grid_executor>
>;


template<class ExecutionAgentTraits, class Function>
struct execute_agent_functor
{
  __host__ __device__
  execute_agent_functor(const typename ExecutionAgentTraits::param_type &param,
                        const Function& f)
    : param_(param),
      f_(f)
  {}

  template<class ExecutorIndex, class SharedParam>
  __device__
  void operator()(ExecutorIndex agent_idx, SharedParam&& shared_params) const
  {
    ExecutionAgentTraits::execute(agent_idx, param_, f_, shared_params);
  }

  template<class ExecutorIndex, class SharedParam>
  __device__
  void operator()(ExecutorIndex agent_idx, SharedParam&& shared_params)
  {
    ExecutionAgentTraits::execute(agent_idx, param_, f_, shared_params);
  }

  typename ExecutionAgentTraits::param_type param_;
  Function f_;
};


template<class ExecutionAgentTraits, class Function>
__host__ __device__
execute_agent_functor<ExecutionAgentTraits,Function>
  make_execute_agent_functor(const typename ExecutionAgentTraits::param_type& param,
                             const Function& f)
{
  return execute_agent_functor<ExecutionAgentTraits,Function>(param,f);
}


template<class Function>
void bulk_invoke(__cuda::execution_policy& exec, Function&& f)
{
  using execution_agent_type = typename __cuda::execution_policy::execution_agent_type;
  using traits = std::execution_agent_traits<execution_agent_type>;

  auto param = exec.param();
  auto execute_me = make_execute_agent_functor<traits>(param, f);
  auto shape = traits::shape(param);
  auto shared_init = traits::make_shared_initializer(param);

  using executor_type = typename __cuda::execution_policy::executor_type;

  return executor_traits<executor_type>::bulk_invoke(exec.executor(), execute_me, shape, shared_init);
}


} // end __cuda


template<class Function, class... Args>
void bulk_invoke(__cuda::execution_policy&& exec, Function&& f, Args&&... args)
{
  auto g = thrust::experimental::bind(f, thrust::placeholders::_1, std::forward<Args>(args)...);
  __cuda::bulk_invoke(exec, g);
}


template<class Function, class... Args>
std::future<void> bulk_async(__cuda::execution_policy&& exec, Function&& f, Args&&... args)
{
  std::cout << "bulk_async(__cuda::execution_policy)" << std::endl;
  return std::make_ready_future();
}


}

