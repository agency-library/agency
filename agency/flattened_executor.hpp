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
    using index_type = detail::flattened_shape_type_t<typename base_traits::index_type>;

    template<class T>
    using future = typename base_traits::template future<T>;

    template<class T>
    using container = detail::array<T, shape_type, typename base_traits::template allocator<T>>;

    future<void> make_ready_future()
    {
      return executor_traits<base_executor_type>::make_ready_future(base_executor());
    }

    flattened_executor(const base_executor_type& base_executor = base_executor_type())
      : min_inner_size_(1000),
        outer_subscription_(std::max(1u, log2(std::max(1u,std::thread::hardware_concurrency())))),
        base_executor_(base_executor)
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

    // the type of shape_type's head is simply its first element when the shape_type is a tuple
    // otherwise, it's just the shape_type
    using shape_head_type = typename std::decay<
      decltype(detail::tuple_head_if(std::declval<shape_type>()))
    >::type;

    // the type of shape_type's tail is its tail when the shape_type is a tuple
    // otherwise, it's just an empty tuple
    using shape_tail_type = decltype(detail::tuple_tail_if(std::declval<shape_type>()));

    // when shape_type is a tuple, returns a reference to its first element
    // otherwise, returns shape
    static const shape_head_type& shape_head(const shape_type& shape)
    {
      return detail::tuple_head_if(shape);
    }

    // when shape_type is a tuple, returns its tail
    // otherwise, returns an empty tuple
    static shape_tail_type shape_tail(const shape_type& shape)
    {
      return detail::tuple_tail_if(shape);
    }

    using partition_type = detail::tuple<
      typename std::tuple_element<0,base_shape_type>::type,
      typename std::tuple_element<1,base_shape_type>::type
    >;

    // returns (outer size, inner size)
    partition_type partition_head(const shape_head_type& shape) const
    {
      // avoid division by zero below
      // XXX implement me!
//      if(is_empty(shape)) return partition_type{};

      using outer_shape_type = typename std::tuple_element<0,base_shape_type>::type;
      using inner_shape_type = typename std::tuple_element<1,base_shape_type>::type;

      // XXX this aritmetic assumes that outer_shape_type and inner_shape_type are scalar instead of tuples
      outer_shape_type outer_size = (shape + min_inner_size_ - 1) / min_inner_size_;

      outer_size = std::min<size_t>(outer_subscription_ * std::thread::hardware_concurrency(), outer_size);

      inner_shape_type inner_size = (shape + outer_size - 1) / outer_size;

      return partition_type{outer_size, inner_size};
    }

    template<size_t... Indices>
    static base_shape_type make_base_shape_impl(detail::index_sequence<Indices...>, const partition_type& partition_of_head, const shape_tail_type& tail)
    {
      return base_shape_type{detail::get<0>(partition_of_head), detail::get<1>(partition_of_head), detail::get<Indices>(tail)...};
    }

    static base_shape_type make_base_shape(const partition_type& partition_of_head, const shape_tail_type& tail)
    {
      auto indices = detail::make_index_sequence<std::tuple_size<shape_tail_type>::value>();

      return make_base_shape_impl(indices, partition_of_head, tail);
    }

    base_shape_type partition_into_base_shape(const shape_type& shape) const
    {
      // split the shape into its head and tail
      shape_head_type head = shape_head(shape);
      shape_tail_type tail = shape_tail(shape);

      // to partition the head and tail into the base_shape_type,
      // we partition the head and then concatenate the resulting tuple with the tail
      return make_base_shape(partition_head(head), tail);
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

