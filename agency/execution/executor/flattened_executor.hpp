#pragma once

#include <type_traits>
#include <agency/detail/tuple.hpp>
#include <agency/detail/factory.hpp>
#include <agency/detail/array.hpp>
#include <agency/detail/shape.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/execution_categories.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/executor/scoped_executor.hpp>
#include <agency/execution/executor/detail/flatten_index_and_invoke.hpp>
#include <agency/execution/executor/detail/customization_points/bulk_continuation_executor_adaptor.hpp>

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

  using container_type = detail::result_of_t<Factory(Shape)>;

  __agency_exec_check_disable__
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
struct flattened_execution_tag_impl<scoped_execution_tag<OuterCategory,InnerCategory>>
{
  using type = parallel_execution_tag;
};

template<class OuterCategory, class InnerCategory, class InnerInnerCategory>
struct flattened_execution_tag_impl<scoped_execution_tag<OuterCategory, scoped_execution_tag<InnerCategory,InnerInnerCategory>>>
{
  // OuterCategory and InnerInnerCategory merge into parallel as the outer category
  // while InnerInnerCategory is promoted to the inner category
  using type = scoped_execution_tag<
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
  // probably shouldn't insist on a scoped executor
  static_assert(
    detail::is_scoped_execution_category<typename executor_traits<Executor>::execution_category>::value,
    "Execution category of Executor must be scoped."
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

    flattened_executor(const base_executor_type& base_executor = base_executor_type())
      : base_executor_(base_executor)
    {}

    template<class Function, class Factory1, class Future, class Factory2, class... Factories,
             class = typename std::enable_if<
               sizeof...(Factories) == execution_depth - 1
             >::type>
    future<detail::result_of_t<Factory1(shape_type)>>
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
      using result_type = detail::result_of_t<Factory1(shape_type)>;
      return executor_traits<base_executor_type>::template future_cast<result_type>(base_executor(), intermediate_fut);
    }


    template<class Function, class Future, class ResultFactory, class OuterFactory, class... InnerFactories,
             __AGENCY_REQUIRES(sizeof...(InnerFactories) == execution_depth - 1)
            >
    future<detail::result_of_t<ResultFactory()>>
      bulk_then_execute(Function f, shape_type shape, Future& predecessor, ResultFactory result_factory, OuterFactory outer_factory, InnerFactories... inner_factories)
    {
      base_shape_type base_shape = partition_into_base_shape(shape);

      using base_index_type = detail::executor_index_t<base_executor_type>;
      using future_value_type = detail::future_value_t<Future>;
      auto execute_me = detail::make_new_flatten_index_and_invoke<base_index_type,future_value_type>(f, base_shape, shape);

      experimental::bulk_continuation_executor_adaptor<base_executor_type> adapted_executor(base_executor());

      return adapted_executor.bulk_then_execute(execute_me, base_shape, predecessor, result_factory, outer_factory, agency::detail::unit_factory(), inner_factories...);
    }

    const base_executor_type& base_executor() const
    {
      return base_executor_;
    }

    base_executor_type& base_executor()
    {
      return base_executor_;
    }

    shape_type shape() const
    {
      // to flatten the base executor's shape we merge the two front dimensions together
      return detail::merge_front_shape_elements(base_traits::shape(base_executor()));
    }

    shape_type max_shape_dimensions() const
    {
      // to flatten the base executor's shape we merge the two front dimensions together
      return detail::merge_front_shape_elements(base_traits::max_shape_dimensions(base_executor()));
    }

  private:
    using base_shape_type = typename base_traits::shape_type;

    static_assert(detail::is_tuple<base_shape_type>::value, "The shape_type of flattened_executor's base_executor must be a tuple.");

    using shape_head_type = detail::shape_head_t<shape_type>;

    using shape_tail_type = detail::shape_tail_t<shape_type>;

    using head_partition_type = detail::tuple<
      typename std::tuple_element<0,base_shape_type>::type,
      typename std::tuple_element<1,base_shape_type>::type
    >;

    head_partition_type partition_head(const shape_head_type& shape) const
    {
      // avoid division by zero outer_size below
      size_t size = detail::shape_cast<size_t>(shape);
      if(size == 0)
      {
        return head_partition_type{};
      }

      base_shape_type base_executor_shape = base_traits::shape(base_executor());
      size_t outer_granularity = detail::shape_head_size(base_executor_shape);
      size_t inner_granularity = detail::shape_size(detail::get<1>(base_executor_shape));

      base_shape_type base_executor_max_sizes = detail::max_sizes(base_traits::max_shape_dimensions(base_executor()));

      size_t outer_max_size = detail::shape_head(base_executor_max_sizes);
      size_t inner_max_size = detail::get<1>(base_executor_max_sizes);

      // set outer subscription to 1
      size_t outer_size = std::min(outer_max_size, std::min(size, outer_granularity));

      size_t inner_size = (size + outer_size - 1) / outer_size;

      // address inner underutilization
      // XXX consider trying to balance the utilization
      while(inner_size < inner_granularity)
      {
        // halve the outer size
        outer_size = std::max<int>(1, outer_size / 2);
        inner_size *= 2;
      }

      if(inner_size > inner_max_size)
      {
        inner_size = inner_max_size;
        outer_size = (size + inner_size - 1) / inner_size;

        if(outer_size > outer_max_size)
        {
          // XXX we could have asserted size <= outer_max_size * inner_max_size at the very beginning
          throw std::runtime_error("flattened_executor::partition_head(): size is too large to accomodate");
        }
      }

      using outer_shape_type = typename std::tuple_element<0,head_partition_type>::type;
      using inner_shape_type = typename std::tuple_element<1,head_partition_type>::type;

      // XXX we may want to use a different heuristic to lift these sizes into shapes
      //     such as trying to make the shapes as square as possible
      //     or trying to preserve the original aspect ratio of shape somehow
      outer_shape_type outer_shape = detail::shape_cast<outer_shape_type>(outer_size);
      inner_shape_type inner_shape = detail::shape_cast<inner_shape_type>(inner_size);

      return head_partition_type{outer_shape, inner_shape};
    }

    template<size_t... Indices>
    static base_shape_type make_base_shape_impl(detail::index_sequence<Indices...>, const head_partition_type& partition_of_head, const shape_tail_type& tail)
    {
      return base_shape_type{detail::get<0>(partition_of_head), detail::get<1>(partition_of_head), detail::get<Indices>(tail)...};
    }

    static base_shape_type make_base_shape(const head_partition_type& partition_of_head, const shape_tail_type& tail)
    {
      auto indices = detail::make_index_sequence<std::tuple_size<shape_tail_type>::value>();

      return make_base_shape_impl(indices, partition_of_head, tail);
    }

    base_shape_type partition_into_base_shape(const shape_type& shape) const
    {
      // split the shape into its head and tail elements
      shape_head_type head = detail::shape_head(shape);
      shape_tail_type tail = detail::shape_tail(shape);

      // to partition the head and tail elements into the base_shape_type,
      // we partition the head element and then concatenate the resulting tuple with the tail
      return make_base_shape(partition_head(head), tail);
    }

    base_executor_type base_executor_;
};


} // end agency

