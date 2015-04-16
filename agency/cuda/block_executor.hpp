#pragma once

#include <agency/cuda/grid_executor.hpp>
#include <agency/cuda/detail/bind.hpp>
#include <agency/detail/tuple.hpp>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class Function>
struct block_executor_helper_functor
{
  Function f_;

  __device__
  void operator()(grid_executor::index_type idx)
  {
    f_(agency::detail::get<1>(idx));
  }

  template<class T>
  __device__
  void operator()(grid_executor::index_type idx, const agency::detail::ignore_t&, T& inner_shared_param)
  {
    f_(agency::detail::get<1>(idx), inner_shared_param);
  }
};


}


class block_executor : private grid_executor
{
  private:
    using super_t = grid_executor;
    using traits = executor_traits<super_t>;

  public:
    using execution_category = concurrent_execution_tag;

    // XXX probably should be int
    using shape_type = unsigned int;
    using index_type = unsigned int;

    template<class T>
    using future = typename traits::template future<T>;

    using super_t::super_t;
    using super_t::shared_memory_size;
    using super_t::stream;
    using super_t::gpu;
    using super_t::global_function_pointer;

    template<class Function>
    __host__ __device__
    shape_type max_shape(Function f) const
    {
      return super_t::max_shape(f).y;
    }

  public:
    template<class Function>
    future<void> then_execute(future<void>& dependency, Function f, shape_type shape)
    {
      auto g = detail::block_executor_helper_functor<Function>{f};
      return traits::then_execute(dependency, g, super_t::shape_type{1,shape});
    }

    template<class Function>
    future<void> async_execute(Function f, shape_type shape)
    {
      auto g = detail::block_executor_helper_functor<Function>{f};
      return traits::async_execute(g, super_t::shape_type{1,shape});
    }

    template<class Function>
    void execute(Function f, shape_type shape)
    {
      auto g = detail::block_executor_helper_functor<Function>{f};
      traits::execute(g, super_t::shape_type{1,shape});
    }

    template<class Function, class T>
    future<void> async_execute(Function f, shape_type shape, T shared_init)
    {
      auto g = detail::block_executor_helper_functor<Function>{f};
      return traits::async_execute(*this, g, super_t::shape_type{1,shape}, agency::detail::ignore, shared_init);
    }

    template<class Function, class T>
    void execute(Function f, shape_type shape, T shared_init)
    {
      auto g = detail::block_executor_helper_functor<Function>{f};
      traits::execute(*this, g, super_t::shape_type{1,shape}, agency::detail::ignore, shared_init);
    }
};


} // end cuda
} // end agency

