#pragma once

#include <agency/execution_policy.hpp>
#include <agency/flattened_executor.hpp>
#include <type_traits>
#include "execution_agent.hpp"
#include "grid_executor.hpp"
#include "block_executor.hpp"
#include "bind.hpp"


namespace cuda
{
namespace detail
{


template<class Function, class ExecutionAgentTraits>
struct execute_agent_functor
{
  __host__ __device__
  execute_agent_functor(const Function& f,
                        const typename ExecutionAgentTraits::param_type& param)
    : f_(f),
      param_(param)
  {}

  template<class ExecutorIndex, class SharedParam>
  __device__
  void operator()(ExecutorIndex agent_idx, SharedParam&& shared_params) const
  {
    ExecutionAgentTraits::execute(f_, agent_idx, param_, shared_params);
  }

  template<class ExecutorIndex, class SharedParam>
  __device__
  void operator()(ExecutorIndex agent_idx, SharedParam&& shared_params)
  {
    ExecutionAgentTraits::execute(f_, agent_idx, param_, shared_params);
  }


  template<class ExecutorIndex>
  __device__
  void operator()(ExecutorIndex agent_idx, decltype(std::ignore)) const
  {
    ExecutionAgentTraits::execute(f_, agent_idx, param_);
  }

  template<class ExecutorIndex>
  __device__
  void operator()(ExecutorIndex agent_idx, decltype(std::ignore))
  {
    ExecutionAgentTraits::execute(f_, agent_idx, param_);
  }


  template<class ExecutorIndex>
  __device__
  void operator()(ExecutorIndex agent_idx) const
  {
    ExecutionAgentTraits::execute(f_, agent_idx, param_);
  }

  template<class ExecutorIndex>
  __device__
  void operator()(ExecutorIndex agent_idx)
  {
    ExecutionAgentTraits::execute(f_, agent_idx, param_);
  }

  Function f_;
  typename ExecutionAgentTraits::param_type param_;
};


template<class ExecutionAgentTraits, class Function>
__host__ __device__
execute_agent_functor<Function, ExecutionAgentTraits>
  make_execute_agent_functor(const Function& f,const typename ExecutionAgentTraits::param_type& param)
{
  return execute_agent_functor<Function,ExecutionAgentTraits>(f,param);
}


template<class ExecutionPolicy, class Function>
void bulk_invoke(const ExecutionPolicy& exec, Function&& f)
{
  using execution_agent_type = typename ExecutionPolicy::execution_agent_type;
  using traits = agency::execution_agent_traits<execution_agent_type>;

  auto param = exec.param();
  auto execute_me = make_execute_agent_functor<traits>(f, param);
  auto shape = traits::domain(param).shape();
  auto shared_init = traits::make_shared_initializer(param);

  using executor_type = typename ExecutionPolicy::executor_type;

  return agency::executor_traits<executor_type>::bulk_invoke(exec.executor(), execute_me, shape, shared_init);
}


} // end detail


class parallel_execution_policy : public agency::detail::basic_execution_policy<cuda::parallel_agent, agency::flattened_executor<cuda::grid_executor>>
{
  private:
    using super_t = agency::detail::basic_execution_policy<cuda::parallel_agent, agency::flattened_executor<cuda::grid_executor>>;

  public:
    using execution_agent_type = super_t::execution_agent_type;
    using executor_type = super_t::executor_type;
    using param_type = execution_agent_type::param_type;

    using super_t::basic_execution_policy;

    // XXX we shouldn't have to include either of these constructors due to the
    // using above
    parallel_execution_policy() = default;
    parallel_execution_policy(const super_t& other)
      : super_t(other)
    {}

    parallel_execution_policy(const param_type& param, const executor_type& executor = executor_type())
      : super_t(param, executor)
    {}

    parallel_execution_policy(const agency::parallel_execution_policy::param_type& param,
                              const executor_type& executor = executor_type())
      : parallel_execution_policy(param_type(param.domain().min(), param.domain().max()),
                                  executor)
    {}

    template<class Arg1, class... Args>
    parallel_execution_policy operator()(Arg1&& arg1, Args&&... args) const
    {
      return super_t::operator()(std::forward<Arg1>(arg1), std::forward<Args>(args)...);
    }
};


const parallel_execution_policy par{};


class concurrent_execution_policy : public agency::detail::basic_execution_policy<cuda::concurrent_agent, cuda::block_executor>
{
  private:
    using super_t = agency::detail::basic_execution_policy<cuda::concurrent_agent, cuda::block_executor>;

  public:
    using execution_agent_type = super_t::execution_agent_type;
    using executor_type = super_t::executor_type;
    using param_type = execution_agent_type::param_type;

    using super_t::basic_execution_policy;

    concurrent_execution_policy(const param_type& param, const executor_type& executor = executor_type())
      : super_t(param, executor)
    {}

    concurrent_execution_policy(const agency::concurrent_execution_policy::param_type& param,
                                const executor_type& executor = executor_type())
      : concurrent_execution_policy(param_type(param.domain().min(), param.domain().max()),
                                    executor)
    {}
};


template<class Function, class... Args>
void bulk_invoke(const parallel_execution_policy& exec, Function&& f, Args&&... args)
{
  auto g = thrust::experimental::bind(f, thrust::placeholders::_1, std::forward<Args>(args)...);
  detail::bulk_invoke(exec, g);
}


template<class Function, class... Args>
std::future<void> bulk_async(const parallel_execution_policy& exec, Function&& f, Args&&... args)
{
  std::cout << "cuda::bulk_async(cuda::parallel_execution_policy)" << std::endl;
  return agency::detail::make_ready_future();
}


template<class Function, class... Args>
void bulk_invoke(const concurrent_execution_policy& exec, Function&& f, Args&&... args)
{
  auto g = thrust::experimental::bind(f, thrust::placeholders::_1, std::forward<Args>(args)...);
  detail::bulk_invoke(exec, g);
}


template<class Function, class... Args>
std::future<void> bulk_async(const concurrent_execution_policy& exec, Function&& f, Args&&... args)
{
  std::cout << "cuda::bulk_async(cuda::concurrent_execution_policy)" << std::endl;
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
      agency::parallel_execution_tag
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
      agency::concurrent_execution_tag
    >::value
  >::type
>
{
  using type = cuda::concurrent_execution_policy;
};


} // end detail
} // end cuda


namespace agency
{


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


// overload bulk_invoke & bulk_async on cuda's execution policies

template<class Function, class... Args>
void bulk_invoke(cuda::parallel_execution_policy&& exec, Function&& f, Args&&... args)
{
  return cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class Function, class... Args>
void bulk_invoke(cuda::parallel_execution_policy& exec, Function&& f, Args&&... args)
{
  return cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class Function, class... Args>
void bulk_invoke(const cuda::parallel_execution_policy& exec, Function&& f, Args&&... args)
{
  return cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}


template<class Function, class... Args>
std::future<void> bulk_async(cuda::parallel_execution_policy&& exec, Function&& f, Args&&... args)
{
  return cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class Function, class... Args>
std::future<void> bulk_async(cuda::parallel_execution_policy& exec, Function&& f, Args&&... args)
{
  return cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class Function, class... Args>
std::future<void> bulk_async(const cuda::parallel_execution_policy& exec, Function&& f, Args&&... args)
{
  return cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}


template<class Function, class... Args>
void bulk_invoke(cuda::concurrent_execution_policy&& exec, Function&& f, Args&&... args)
{
  return cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class Function, class... Args>
void bulk_invoke(cuda::concurrent_execution_policy& exec, Function&& f, Args&&... args)
{
  return cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class Function, class... Args>
void bulk_invoke(const cuda::concurrent_execution_policy& exec, Function&& f, Args&&... args)
{
  return cuda::bulk_invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}


template<class Function, class... Args>
std::future<void> bulk_async(cuda::concurrent_execution_policy&& exec, Function&& f, Args&&... args)
{
  return cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class Function, class... Args>
std::future<void> bulk_async(cuda::concurrent_execution_policy& exec, Function&& f, Args&&... args)
{
  return cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class Function, class... Args>
std::future<void> bulk_async(const cuda::concurrent_execution_policy& exec, Function&& f, Args&&... args)
{
  return cuda::bulk_async(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}


} // end agency

