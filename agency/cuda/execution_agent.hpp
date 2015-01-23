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
    __device__
    void wait() const
    {
      __syncthreads();
    }

  private:
    struct noop
    {
      __device__ void operator()(super_t&){}
    };

  protected:
    template<class Function>
    __device__
    basic_concurrent_agent(Function f, const typename super_t::index_type& index, const typename super_t::param_type& param)
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

