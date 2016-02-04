#pragma once

#include <type_traits>
#include <agency/detail/tuple.hpp>
#include <agency/executor_traits.hpp>
#include <agency/execution_categories.hpp>
#include <agency/nested_executor.hpp>
#include <agency/detail/factory.hpp>
#include <agency/detail/optional.hpp>
#include <agency/detail/flatten_index_and_invoke.hpp>
#include <agency/detail/array.hpp>

namespace agency
{
namespace detail
{


// XXX we should find a better home for this functionality because grid_executor.hpp replicates this code
template<class Container>
struct guarded_container : Container
{
  using Container::Container;

  __AGENCY_ANNOTATION
  guarded_container()
    : Container()
  {}

  __AGENCY_ANNOTATION
  guarded_container(Container&& other)
    : Container(std::move(other))
  {}

  struct reference
  {
    Container& self_;

    template<class OptionalValueAndIndex>
    __AGENCY_ANNOTATION
    void operator=(OptionalValueAndIndex&& opt)
    {
      if(opt)
      {
        auto idx = opt.value().index;
        self_[idx] = std::forward<OptionalValueAndIndex>(opt).value().value;
      }
    }
  };

  template<class Index>
  __AGENCY_ANNOTATION
  reference operator[](const Index&)
  {
    return reference{*this};
  }
};


template<class Container>
__AGENCY_ANNOTATION
guarded_container<typename std::decay<Container>::type> make_guarded_container(Container&& value)
{
  return guarded_container<typename std::decay<Container>::type>(std::forward<Container>(value));
}


template<class Factory, class Shape>
struct guarded_container_factory
{
  Factory factory_;
  Shape shape_;

  using container_type = typename std::result_of<Factory(Shape)>::type;

  __agency_hd_warning_disable__
  template<class Arg>
  __AGENCY_ANNOTATION
  guarded_container<container_type> operator()(const Arg&)
  {
    return agency::detail::make_guarded_container(factory_(shape_));
  }
};


template<class ExecutionCategory>
struct flattened_execution_tag_impl;

template<class OuterCategory, class InnerCategory>
struct flattened_execution_tag_impl<nested_execution_tag<OuterCategory,InnerCategory>>
{
  using type = parallel_execution_tag;
};

template<class OuterCategory, class InnerCategory, class InnerInnerCategory>
struct flattened_execution_tag_impl<nested_execution_tag<OuterCategory, nested_execution_tag<InnerCategory,InnerInnerCategory>>>
{
  // OuterCategory and InnerInnerCategory merge into parallel as the outer category
  // while InnerInnerCategory is promoted to the inner category
  using type = nested_execution_tag<
    parallel_execution_tag,
    InnerInnerCategory
  >;
};

template<class ExecutionCategory>
using flattened_execution_tag = typename flattened_execution_tag_impl<ExecutionCategory>::type;


template<class Integral>
__AGENCY_ANNOTATION
Integral isqrt_ceil(Integral x)
{
  auto arg = x;

  Integral result = 0;

  // set the second-to-top bit
  Integral one = Integral(1) << ((sizeof(Integral) * 8) - 2);
  
  // "one" starts at the highest power of four <= than the argument.
  while(one > x)
  {
    one >>= 2;
  }
  
  while(one != 0)
  {
    if(x >= result + one)
    {
      x = x - (result + one);
      result = result + 2 * one;
    }

    result >>= 1;
    one >>= 2;
  }

  // if squaring the result doesn't yield arg, then the ceiling is 1 + result
  // XXX could be a better way to do this besides storing arg and doing a multiplication
  if(result * result < arg)
  {
    ++result;
  }
  
  return result;
}


} // end detail


template<class Executor>
class flattened_executor
{
  // probably shouldn't insist on a nested executor
  static_assert(
    detail::is_nested_execution_category<typename executor_traits<Executor>::execution_category>::value,
    "Execution category of Executor must be nested."
  );

  private:
    using base_traits = executor_traits<Executor>;
    using base_execution_category = typename base_traits::execution_category;
    constexpr static auto execution_depth = base_traits::execution_depth - 1;

  public:
    using base_executor_type = Executor;
    using execution_category = detail::flattened_execution_tag<base_execution_category>;
    using shape_type = detail::flattened_shape_type_t<typename base_traits::shape_type>;
    using index_type = detail::flattened_index_type_t<typename base_traits::index_type>;

    template<class T>
    using future = typename base_traits::template future<T>;

    template<class T>
    using allocator = typename base_traits::template allocator<T>;

    template<class T>
    using container = detail::array<T, shape_type, allocator<T>, index_type>;

    future<void> make_ready_future()
    {
      return executor_traits<base_executor_type>::make_ready_future(base_executor());
    }

    flattened_executor(const base_executor_type& base_executor = base_executor_type(),
                       size_t maximum_outer_size = std::numeric_limits<size_t>::max(),
                       size_t maximum_inner_size = std::numeric_limits<size_t>::max())
      :
        maximum_outer_size_(maximum_outer_size),
        maximum_inner_size_(maximum_inner_size),
        base_executor_(base_executor)
    {}

    flattened_executor(size_t maximum_outer_size,
                       size_t maximum_inner_size = std::numeric_limits<size_t>::max())
      : flattened_executor(base_executor_type(), maximum_outer_size, maximum_inner_size)
    {}

    template<class Function, class Factory1, class Future, class Factory2, class... Factories,
             class = typename std::enable_if<
               sizeof...(Factories) == execution_depth - 1
             >::type>
    future<typename std::result_of<Factory1(shape_type)>::type>
      then_execute(Function f, Factory1 result_factory, shape_type shape, Future& dependency, Factory2 outer_factory, Factories... inner_factories)
    {
      base_shape_type base_shape = partition_into_base_shape(shape);

      // store results into an intermediate result
      detail::guarded_container_factory<Factory1,shape_type> intermediate_result_factory{result_factory,shape};

      // create a function to execute
      using base_index_type = typename executor_traits<base_executor_type>::index_type;
      using future_value_type = typename future_traits<Future>::value_type;
      auto execute_me = detail::make_flatten_index_and_invoke<base_index_type,future_value_type>(f, base_shape, shape);


      // then_execute with the base_executor
      auto intermediate_fut = executor_traits<base_executor_type>::then_execute(
        base_executor(),
        execute_me,
        intermediate_result_factory,
        base_shape,
        dependency,
        outer_factory, agency::detail::unit_factory(), inner_factories...
      );

      // cast the intermediate result to the type of result expected by the caller
      using result_type = typename std::result_of<Factory1(shape_type)>::type;
      return executor_traits<base_executor_type>::template future_cast<result_type>(base_executor(), intermediate_fut);
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
    using base_shape_type = typename base_traits::shape_type;

    static_assert(detail::is_tuple<base_shape_type>::value, "The shape_type of flattened_executor's base_executor must be a tuple.");

    // the type of shape_type's last type is simply its last element when the shape_type is a tuple
    // otherwise, it's just the shape_type
    using shape_last_type = typename std::decay<
      decltype(detail::tuple_last_if(std::declval<shape_type>()))
    >::type;

    // the type of shape_type's prefix is simply its prefix when the shape_type is a tuple
    // otherwise, it's just an empty tuple
    using shape_prefix_type = decltype(detail::tuple_prefix_if(std::declval<shape_type>()));

    // when shape_type is a tuple, returns its prefix
    // otherwise, returns an empty tuple
    static shape_prefix_type shape_prefix(const shape_type& shape)
    {
      return detail::tuple_prefix_if(shape);
    }

    // when shape_type is a tuple, returns its last element
    // otherwise, returns shape
    static const shape_last_type& shape_last(const shape_type& shape)
    {
      return detail::tuple_last_if(shape);
    }

    using head_partition_type = detail::tuple<
      typename std::tuple_element<0,base_shape_type>::type,
      typename std::tuple_element<1,base_shape_type>::type
    >;

    using last_partition_type = detail::tuple<
      typename std::tuple_element<std::tuple_size<base_shape_type>::value - 2,base_shape_type>::type,
      typename std::tuple_element<std::tuple_size<base_shape_type>::value - 1,base_shape_type>::type
    >;

    last_partition_type partition_last(const shape_last_type& shape) const
    {
      // avoid division by zero outer_size below
      size_t size = detail::shape_cast<size_t>(shape);
      if(size == 0)
      {
        return last_partition_type{};
      }

      // maximize the inner_size
      size_t inner_size = std::min(size, maximum_inner_size_);

      // divide to find the outer size
      size_t outer_size = (size + inner_size - 1) / inner_size;

      if(outer_size == 0)
      {
        inner_size = detail::isqrt_ceil(size);
        outer_size = (size + inner_size - 1) / inner_size;
      }

      if(outer_size > maximum_outer_size_)
      {
        outer_size = maximum_outer_size_;

        inner_size = (size + maximum_outer_size_ - 1) / outer_size;

        if(inner_size > maximum_inner_size_)
        {
          throw std::runtime_error("flattened_executor::partition_head(): the dimensions of shape are too large");
        }
      }

      using outer_shape_type = typename std::tuple_element<0,last_partition_type>::type;
      using inner_shape_type = typename std::tuple_element<1,last_partition_type>::type;

      // XXX we may want to use a different heuristic to lift these sizes into shapes
      //     such as trying to make the shapes as square as possible
      //     or trying to preserve the original aspect ratio of shape somehow
      outer_shape_type outer_shape = detail::shape_cast<outer_shape_type>(outer_size);
      inner_shape_type inner_shape = detail::shape_cast<inner_shape_type>(inner_size);

      return last_partition_type{outer_shape, inner_shape};
    }


    template<size_t... Indices>
    static base_shape_type make_base_shape_impl(detail::index_sequence<Indices...>, const shape_prefix_type& prefix, const last_partition_type& partition_of_last)
    {
      return base_shape_type{detail::get<Indices>(prefix)..., detail::get<0>(partition_of_last), detail::get<1>(partition_of_last)};
    }

    static base_shape_type make_base_shape(const shape_prefix_type& prefix, const last_partition_type& partition_of_last)
    {
      auto indices = detail::make_index_sequence<std::tuple_size<shape_prefix_type>::value>();

      return make_base_shape_impl(indices, prefix, partition_of_last);
    }

    base_shape_type partition_into_base_shape(const shape_type& shape) const
    {
      // split the shape into its prefix and last element
      shape_prefix_type prefix = shape_prefix(shape);
      shape_last_type last = shape_last(shape);

      // to partition the prefix and last element into the base_shape_type,
      // we partition the last element and then concatenate the resulting tuple with the prefix
      return make_base_shape(prefix, partition_last(last));
    }

    size_t maximum_outer_size_;
    size_t maximum_inner_size_;
    base_executor_type base_executor_;
};


} // end agency

