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

    __host__ __device__
    size_t index() const
    {
      return index_;
    }

    __host__ __device__
    const detail::domain& domain() const
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
        __host__ __device__
        param_type(const param_type& other)
          : domain_(other.domain_)
        {}

        __host__ __device__
        param_type(const detail::domain& d)
          : domain_(d)
        {}

        __host__ __device__
        param_type(size_t min, size_t max)
          : domain_(min,max)
        {}

        __host__ __device__
        const detail::domain& domain() const
        {
          return domain_;
        }

      private:
        detail::domain domain_;
    };

  protected:
    template<class Function, class Index>
    __host__ __device__
    basic_execution_agent_base(Function f, const Index& index, const param_type& param)
      : index_(index),
        domain_(param.domain())
    {
      f(*this);
    }

  private:
    size_t index_;
    detail::domain domain_;
};


template<class ExecutionCategory>
class basic_execution_agent : public basic_execution_agent_base<ExecutionCategory>
{
  private:
    using super_t = basic_execution_agent_base<ExecutionCategory>;

  public:
    using basic_execution_agent_base<ExecutionCategory>::basic_execution_agent_base;
    using param_type = typename basic_execution_agent_base<ExecutionCategory>::param_type;

    friend struct std::execution_agent_traits<basic_execution_agent>;

    struct noop
    {
      template<class T>
      __host__ __device__
      void operator()(T&){}
    };

  protected:
    template<class Function, class Index>
    __host__ __device__
    basic_execution_agent(Function f, const Index& index, const param_type& param)
      : super_t(noop(), index, param)
    {
      f(*this);
    }
};


} // end detail


using parallel_agent = detail::basic_execution_agent<std::parallel_execution_tag>;


} // end cuda


// specialize execution_traits on cuda::parallel_agent
namespace std
{


template<>
struct execution_agent_traits<cuda::parallel_agent>
  : std::execution_agent_traits<cuda::detail::basic_execution_agent_base<std::parallel_execution_tag>>
{
  template<class Function, class Tuple>
  __host__ __device__
  static typename enable_if<
    (__tuple_size_if_tuple_else_zero<shape_type>::value == __tuple_size_if_tuple_else_zero<Tuple>::value)
  >::type
    execute(Function f, const Tuple& indices, const param_type& param)
  {
    cuda::parallel_agent agent(f, indices, param);
  }
};


} // end std

