#pragma once

#include <cstddef>
#include <execution_agent>

namespace cuda
{
namespace detail
{


struct domain
{
  // XXX might want to make this int for CUDA
  using value_type = std::size_t;
  using size_type  = std::size_t;

  __host__ __device__
  domain()
    : min_{}, max_{}
  {}

  __host__ __device__
  domain(const value_type& size)
    : min_(0), max_(size)
  {}

  __host__ __device__
  domain(const value_type& min, const value_type& max)
    : min_(min), max_(max)
  {}

  __host__ __device__
  value_type min() const
  {
    return min_;
  }

  __host__ __device__
  value_type max() const
  {
    return max_;
  }

  value_type min_, max_;

  __host__ __device__
  value_type operator[](size_t idx) const
  {
    return min() + idx;
  }

  __host__ __device__
  size_type size() const
  {
    return max() - min();
  }

  __host__ __device__
  size_type shape() const
  {
    return size();
  }
};


// XXX provide a base class for __basic_execution_agent
//     so that we can use it to derive specializations
//     of std::execution_agent_traits
template<class ExecutionCategory>
class basic_execution_agent_base
{
  public:
    using execution_category = ExecutionCategory;

    // XXX might want to make index return int for CUDA
    using index_type = size_t;

    __host__ __device__
    index_type index() const
    {
      return index_;
    }

    using domain_type = detail::domain;

    __host__ __device__
    const domain_type& domain() const
    {
      return domain_;
    }

    // XXX should provide group_shape(), not group_size()
    //     group_size() is derivable from group_shape
    //     group_shape() is only derivable from group_size()
    //     for 1D domains
    __host__ __device__
    size_t group_size() const
    {
      return domain_.size();
    }

    class param_type
    {
      public:
        param_type() = default;

        __host__ __device__
        param_type(const param_type& other)
          : domain_(other.domain_)
        {}

        __host__ __device__
        param_type(const domain_type& d)
          : domain_(d)
        {}

        __host__ __device__
        param_type(index_type min, index_type max)
          : param_type(domain_type(min,max))
        {}

        __host__ __device__
        const domain_type& domain() const
        {
          return domain_;
        }

      private:
        domain_type domain_;
    };

    static domain_type domain(const param_type& p)
    {
      return p.domain();
    }

  protected:
    template<class Function>
    __host__ __device__
    basic_execution_agent_base(Function f, const index_type& index, const param_type& param)
      : index_(index),
        domain_(param.domain())
    {
      f(*this);
    }

  private:
    index_type index_;
    detail::domain domain_;
};


template<class ExecutionCategory>
class basic_execution_agent : public basic_execution_agent_base<ExecutionCategory>
{
  private:
    using super_t = basic_execution_agent_base<ExecutionCategory>;

  public:
    using super_t::basic_execution_agent_base;
    using typename super_t::index_type;
    using param_type = typename super_t::param_type;

    friend struct std::execution_agent_traits<basic_execution_agent>;

    struct noop
    {
      template<class T>
      __host__ __device__
      void operator()(T&){}
    };

  protected:
    template<class Function>
    __host__ __device__
    basic_execution_agent(Function f, const index_type& index, const param_type& param)
      : super_t(noop(), index, param)
    {
      f(*this);
    }
};


} // end detail


using parallel_agent = detail::basic_execution_agent<std::parallel_execution_tag>;


class concurrent_agent : public detail::basic_execution_agent<std::concurrent_execution_tag>
{
  private:
    using super_t = detail::basic_execution_agent<std::concurrent_execution_tag>;

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
      shared_param_type(const typename super_t::param_type& param)
 //       : count_(param.domain().size()),
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
//      std::barrier barrier_;
    };

    __host__ __device__
    static shared_param_type make_shared_initializer(const param_type& param)
    {
      return shared_param_type(param);
    }

  private:
//    std::barrier &barrier_;
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

    // friend std::execution_agent_traits to give it access to the constructor
    friend struct std::execution_agent_traits<concurrent_agent>;
};


} // end cuda


// specialize execution_traits on cuda::parallel_agent
namespace std
{


template<>
struct execution_agent_traits<cuda::parallel_agent>
  : std::execution_agent_traits<cuda::detail::basic_execution_agent_base<std::parallel_execution_tag>>
{
  template<class Function>
  __host__ __device__
  static void execute(Function f, const index_type& index, const param_type& param)
  {
    cuda::parallel_agent agent(f, index, param);
  }
};


template<>
struct execution_agent_traits<cuda::concurrent_agent>
  : std::execution_agent_traits<cuda::detail::basic_execution_agent_base<std::concurrent_execution_tag>>
{
  template<class Function, class Tuple>
  __host__ __device__
  static void execute(Function f, const index_type& index, const param_type& param, Tuple& shared_params)
  {
    cuda::concurrent_agent agent(f, index, param, shared_params);
  }

  using has_make_shared_initializer = std::true_type;

  __host__ __device__
  static cuda::concurrent_agent::shared_param_type make_shared_initializer(const cuda::concurrent_agent::param_type& param)
  {
    return cuda::concurrent_agent::make_shared_initializer(param);
  }
};


} // end std

