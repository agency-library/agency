#pragma once

#include <cstddef>
#include <agency/execution_agent.hpp>

namespace agency
{
namespace cuda
{


using parallel_agent = parallel_agent;


class concurrent_agent : public agency::detail::basic_execution_agent<concurrent_execution_tag>
{
  private:
    using super_t = agency::detail::basic_execution_agent<concurrent_execution_tag>;

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
    static shared_param_type make_shared_initializer(const param_type& param)
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
    template<class Function, class Index>
    __host__ __device__
    concurrent_agent(Function f, const Index& index, const param_type& param, shared_param_type& shared_param)
//      : super_t(noop(), index, param),
//        barrier_(shared_param.barrier_)
      : super_t(noop(), index, param)
    {
      f(*this);
    }

    // friend agency::execution_agent_traits to give it access to the constructor
    friend struct execution_agent_traits<concurrent_agent>;
};


} // end cuda
} // end agency

