#pragma once

#include <type_traits>
#include <agency/detail/tuple.hpp>
#include <agency/executor_traits.hpp>
#include <agency/execution_categories.hpp>
#include <agency/nested_executor.hpp>

namespace agency
{


template<class Executor>
class flattened_executor
{
  // probably shouldn't insist on a nested executor
  static_assert(
    detail::is_nested_execution_category<typename executor_traits<Executor>::execution_category>::value,
    "Execution category of Executor must be nested."
  );

  public:
    // XXX what is the execution category of a flattened executor?
    using execution_category = parallel_execution_tag;

    using base_executor_type = Executor;

    // XXX maybe use whichever of the first two elements of base_executor_type::shape_type has larger dimensionality?
    using shape_type = size_t;

    template<class T>
    using future = typename executor_traits<Executor>::template future<T>;

    flattened_executor(const base_executor_type& base_executor = base_executor_type())
      : min_inner_size_(1000),
        outer_subscription_(std::max(1u, log2(std::min(1u,std::thread::hardware_concurrency())))),
        base_executor_(base_executor)
    {}

    template<class Function, class T>
    future<void> bulk_async(Function f, shape_type shape, T shared_arg)
    {
      return bulk_async_impl(base_executor(), f, shape, shared_arg);
    }

    const base_executor_type& base_executor() const
    {
      return base_executor_;
    }

    base_executor_type& base_executor()
    {
      return base_executor_;
    }

  private:
    template<class OtherExecutor, class Function, class T>
    future<void> bulk_async_impl(OtherExecutor& exec, Function f, shape_type shape, T shared_arg)
    {
      auto partitioning = partition(shape);

      auto shared_init = detail::make_tuple(shared_arg, agency::detail::ignore);
      using shared_param_type = typename executor_traits<OtherExecutor>::template shared_param_type<decltype(shared_init)>;
      using index_type = typename executor_traits<OtherExecutor>::index_type;

      return executor_traits<OtherExecutor>::bulk_async(exec, [=](index_type idx, shared_param_type shared_params)
      {
        auto flat_idx = agency::detail::get<0>(idx) * agency::detail::get<1>(partitioning) + agency::detail::get<1>(idx);

        if(flat_idx < shape)
        {
          f(flat_idx, agency::detail::get<0>(shared_params));
        }
      },
      partitioning,
      shared_init
      );
    }

    // we can avoid the if(flat_idx < shape) branch above by providing a specialization for nested_executor
    template<class OuterExecutor, class InnerExecutor, class Function, class T>
    future<void> bulk_async_impl(nested_executor<OuterExecutor,InnerExecutor>& exec, Function f, shape_type shape, T shared_arg)
    {
      auto partitioning = partition(shape);

      using shared_param_type = typename executor_traits<OuterExecutor>::template shared_param_type<T>;

      using outer_index_type = typename executor_traits<OuterExecutor>::index_type;
    
      return executor_traits<OuterExecutor>::bulk_async(exec.outer_executor(), [=,&exec](outer_index_type outer_idx, shared_param_type shared_arg)
      {
        auto subgroup_begin = outer_idx * agency::detail::get<1>(partitioning);
        auto subgroup_end   = std::min(shape, subgroup_begin + agency::detail::get<1>(partitioning));

        using inner_index_type = typename executor_traits<InnerExecutor>::index_type;

        executor_traits<InnerExecutor>::bulk_invoke(exec.inner_executor(), [=,&shared_arg](inner_index_type inner_idx)
        {
          auto index = subgroup_begin + inner_idx;
    
          f(index, shared_arg);
        },
        subgroup_end - subgroup_begin);
      },
      agency::detail::get<0>(partitioning),
      shared_arg);
    }

    using partition_type = typename executor_traits<base_executor_type>::shape_type;

    // returns (outer size, inner size)
    partition_type partition(shape_type shape) const
    {
      // avoid division by zero below
      // XXX implement me!
//      if(is_empty(shape)) return partition_type{};

      using outer_shape_type = typename std::tuple_element<0,partition_type>::type;
      using inner_shape_type = typename std::tuple_element<1,partition_type>::type;

      outer_shape_type outer_size = (shape + min_inner_size_ - 1) / min_inner_size_;

      outer_size = std::min<size_t>(outer_subscription_ * std::thread::hardware_concurrency(), outer_size);

      inner_shape_type inner_size = (shape + outer_size - 1) / outer_size;

      return partition_type{outer_size, inner_size};
    }

    inline static unsigned int log2(unsigned int x)
    {
      unsigned int result = 0;
      while(x >>= 1) ++result;
      return result;
    }

    size_t min_inner_size_;
    size_t outer_subscription_;
    base_executor_type base_executor_;
};


} // end agency

