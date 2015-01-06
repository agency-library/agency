#pragma once

#include <cstddef>
#include <agency/execution_agent.hpp>

namespace agency
{
namespace cuda
{


using parallel_agent    = agency::parallel_agent;
using parallel_agent_1d = agency::parallel_agent_1d;
using parallel_agent_2d = agency::parallel_agent_2d;


namespace detail
{


template<class Index>
class basic_concurrent_agent : public agency::detail::basic_execution_agent<concurrent_execution_tag, Index>
{
  private:
    using super_t = agency::detail::basic_execution_agent<concurrent_execution_tag, Index>;

  public:
    __host__ __device__
    void wait() const
    {
      #ifndef __CUDA_ARCH__
//        barrier_.count_down_and_wait();
      #else
        __syncthreads();
      #endif
    }

    struct shared_param_type
    {
      __host__ __device__
      shared_param_type(const typename super_t::param_type& param) //       : count_(param.domain().size()),
 //         barrier_(count_)
        : count_(param.domain().size())
      {}

      __host__ __device__
      shared_param_type(const shared_param_type& other)
//        : count_(other.count_),
//          barrier_(count_)
        : count_(other.count_)
      {}

      int count_;
//      agency::barrier barrier_;
    };

    __host__ __device__
    static shared_param_type make_shared_initializer(const typename super_t::param_type& param)
    {
      return shared_param_type(param);
    }

  private:
//    agency::barrier &barrier_;
//

    struct noop
    {
      __host__ __device__ void operator()(super_t&){}
    };

  protected:
    template<class Function>
    __host__ __device__
    basic_concurrent_agent(Function f, const typename super_t::index_type& index, const typename super_t::param_type& param, shared_param_type& shared_param)
//      : super_t(noop(), index, param),
//        barrier_(shared_param.barrier_)
      : super_t(noop(), index, param)
    {
      f(*this);
    }

    // friend agency::execution_agent_traits to give it access to the constructor
    friend struct execution_agent_traits<basic_concurrent_agent>;
};


} // end detail


using concurrent_agent = detail::basic_concurrent_agent<size_t>;
using concurrent_agent_1d = concurrent_agent;
using concurrent_agent_2d = detail::basic_concurrent_agent<size2>;


} // end cuda
} // end agency

