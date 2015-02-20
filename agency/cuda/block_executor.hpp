#pragma once

#include <agency/cuda/grid_executor.hpp>
#include <agency/cuda/detail/bind.hpp>
#include <agency/detail/ignore.hpp>

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

  template<class Tuple>
  __device__
  void operator()(grid_executor::index_type idx, Tuple&& shared_params)
  {
    f_(agency::detail::get<1>(idx), thrust::get<1>(shared_params));
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
    future<void> bulk_async(Function f, shape_type shape)
    {
      auto g = detail::block_executor_helper_functor<Function>{f};
      return traits::bulk_async(g, super_t::shape_type{1,shape});
    }

    template<class Function>
    void bulk_invoke(Function f, shape_type shape)
    {
      auto g = detail::block_executor_helper_functor<Function>{f};
      traits::bulk_invoke(g, super_t::shape_type{1,shape});
    }

    template<class Function, class T>
    future<void> bulk_async(Function f, shape_type shape, T shared_arg)
    {
      auto g = detail::block_executor_helper_functor<Function>{f};
      return traits::bulk_async(*this, g, super_t::shape_type{1,shape}, thrust::make_tuple(agency::detail::ignore, shared_arg));
    }

    template<class Function, class T>
    void bulk_invoke(Function f, shape_type shape, T shared_arg)
    {
      auto g = detail::block_executor_helper_functor<Function>{f};
      traits::bulk_invoke(*this, g, super_t::shape_type{1,shape}, thrust::make_tuple(agency::detail::ignore, shared_arg));
    }
};


template<class Function, class... Args>
__host__ __device__
void bulk_invoke(block_executor& ex, typename grid_executor::shape_type shape, Function&& f, Args&&... args)
{
  auto g = detail::bind(std::forward<Function>(f), detail::placeholders::_1, std::forward<Args>(args)...);
  ex.bulk_invoke(g, shape);
}


} // end cuda
} // end agency

