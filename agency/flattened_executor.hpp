#pragma once

#include <type_traits>
#include <agency/detail/tuple.hpp>
#include <agency/executor_traits.hpp>
#include <agency/execution_categories.hpp>
#include <agency/nested_executor.hpp>
#include <agency/detail/factory.hpp>
#include <agency/detail/optional.hpp>
#include <agency/detail/project_index_and_invoke.hpp>
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
    return make_guarded_container(factory_(shape_));
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


template<class TypeList>
struct flattened_shape_type;

template<class Shape1, class Shape2, class... Shapes>
struct flattened_shape_type<type_list<Shape1,Shape2,Shapes...>>
{
  // XXX we probably want to think carefully about what it means two "merge" two arithmetic tuples together
  template<class T1, class T2>
  using merge_shapes_t = typename std::common_type<T1,T2>::type;

  using tuple_type = shape_tuple<
    merge_shapes_t<Shape1,Shape2>,
    Shapes...
  >;

  // unwrap single-element tuples
  using type = typename std::conditional<
    (std::tuple_size<tuple_type>::value == 1),
    typename std::tuple_element<0,tuple_type>::type,
    tuple_type
  >::type;
};


// XXX need to do something similar as execution_category
//     the two leftmost indices merge into one while the rest of the indices shift one space left
template<class ShapeTuple>
using flattened_shape_type_t = typename flattened_shape_type<tuple_elements<ShapeTuple>>::type;


template<class TypeList>
struct flattened_index_type;

template<class Index1, class Index2, class... Indices>
struct flattened_index_type<type_list<Index1,Index2,Indices...>>
{
  // XXX we probably want to think carefully about what it means two "merge" two arithmetic tuples together
  template<class T1, class T2>
  using merge_indices_t = typename std::common_type<T1,T2>::type;

  using tuple_type = index_tuple<
    merge_indices_t<Index1,Index2>,
    Indices...
  >;

  // unwrap single-element tuples
  using type = typename std::conditional<
    (std::tuple_size<tuple_type>::value == 1),
    typename std::tuple_element<0,tuple_type>::type,
    tuple_type
  >::type;
};

template<class IndexTuple>
using flattened_index_type_t = flattened_shape_type_t<IndexTuple>;


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

    template<class Function, class Factory1, class Future, class Factory2>
    future<typename std::result_of<Factory1(shape_type)>::type>
      then_execute(Function f, Factory1 result_factory, shape_type shape, Future& dependency, Factory2 shared_factory)
    {
      auto partitioning = partition(shape);

      // store results into an intermediate result
      detail::guarded_container_factory<Factory1,shape_type> intermediate_result_factory{result_factory,shape};

      // create a function to execute
      using base_index_type = typename executor_traits<base_executor_type>::index_type;
      auto execute_me = detail::make_project_index_and_invoke<base_index_type>(f, partitioning, shape);

      // execute
      auto intermediate_fut = executor_traits<base_executor_type>::then_execute(base_executor(), execute_me, intermediate_result_factory, partitioning, dependency, shared_factory, agency::detail::unit_factory());

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

