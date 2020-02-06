#pragma once

#include <type_traits>
#include <agency/tuple.hpp>
#include <agency/detail/factory.hpp>
#include <agency/experimental/ndarray.hpp>
#include <agency/detail/shape.hpp>
#include <agency/detail/index.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/scoped_in_place_type.hpp>
#include <agency/execution/executor/executor_traits/detail/member_barrier_type_or.hpp>
#include <agency/execution/executor/executor_traits/detail/is_scoped_executor.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/executor/scoped_executor.hpp>
#include <agency/execution/executor/customization_points.hpp>
#include <agency/execution/executor/properties/bulk_guarantee.hpp>
#include <agency/execution/executor/query.hpp>
#include <agency/detail/algorithm/min.hpp>
#include <agency/detail/algorithm/max.hpp>


namespace agency
{
namespace detail
{


// XXX this should flatten to parallel_t only if neither OuterGuarantee nor InnerGuarantee are unsequenced
template<class OuterGuarantee, class InnerGuarantee>
__AGENCY_ANNOTATION
constexpr bulk_guarantee_t::parallel_t flatten_bulk_guarantee(bulk_guarantee_t::scoped_t<OuterGuarantee, InnerGuarantee>)
{
  return bulk_guarantee_t::parallel_t{};
}


template<class OuterGuarantee, class InnerGuarantee, class InnerInnerGuarantee>
__AGENCY_ANNOTATION
constexpr auto flatten_bulk_guarantee(bulk_guarantee_t::scoped_t<OuterGuarantee, bulk_guarantee_t::scoped_t<InnerGuarantee, InnerInnerGuarantee>> guarantee) ->
  decltype(
    bulk_guarantee_t::scoped(
      detail::flatten_bulk_guarantee(
        bulk_guarantee_t::scoped(
          guarantee.outer(), guarantee.inner().outer()
        )
      ),
      guarantee.inner().inner()
    )
  )
{
  // OuterGuarantee and InnerGuarantee get scoped and flattened as the outer guarantee
  // while InnerInnerGuarantee is promoted to the inner guarantee

  return bulk_guarantee_t::scoped(
    detail::flatten_bulk_guarantee(
      bulk_guarantee_t::scoped(
        guarantee.outer(), guarantee.inner().outer()
      )
    ),
    guarantee.inner().inner()
  );
}


template<class ShapeTuple>
using flattened_shape_type_t = merge_front_shape_elements_t<ShapeTuple>;


// XXX might not want to use a alias template here
template<class IndexTuple>
using flattened_index_type_t = merge_front_shape_elements_t<IndexTuple>;


template<class ScopedInPlaceBarrier>
struct flattened_barrier_type;

template<>
struct flattened_barrier_type<void>
{
  using type = void;
};

template<class OuterBarrier, class InnerBarrier>
struct flattened_barrier_type<scoped_in_place_type_t<OuterBarrier,InnerBarrier>>
{
  // the InnerBarrier is elided
  using type = OuterBarrier;
};

template<class OuterBarrier, class InnerBarrier, class... Barriers>
struct flattened_barrier_type<scoped_in_place_type_t<OuterBarrier,InnerBarrier,Barriers...>>
{
  // OuterBarrier and InnerBarrier are replaced by OuterBarrier
  // while the innermost Barriers are promoted one scope
  using type = scoped_in_place_type_t<
    OuterBarrier,
    Barriers...
  >;
};

template<class Barrier>
using flattened_barrier_type_t = typename flattened_barrier_type<Barrier>::type;


// flatten_index_and_invoke is used by flattened_executor::bulk_then_execute()
// this definition is for the general case when the predecessor future's type is non-void
template<class Index, class Predecessor, class Function, class Shape>
struct flatten_index_and_invoke
{
  using index_type = Index;
  using shape_type = Shape;

  using flattened_index_type = flattened_index_type_t<Index>;
  using flattened_shape_type = flattened_shape_type_t<Shape>;

  mutable Function     f_;
  shape_type           shape_;
  flattened_shape_type flattened_shape_;

  __AGENCY_ANNOTATION
  flatten_index_and_invoke(const Function& f, shape_type shape, flattened_shape_type flattened_shape)
    : f_(f),
      shape_(shape),
      flattened_shape_(flattened_shape)
  {}

  __AGENCY_ANNOTATION
  flattened_index_type flatten_index(const Index& idx) const
  {
    return detail::merge_front_index_elements(idx, shape_);
  }

  __AGENCY_ANNOTATION
  bool in_domain(const flattened_index_type& idx) const
  {
    // idx is in the domain of f_ if idx is contained within the
    // axis-aligned bounded box from extremal corners at the origin
    // and flattened_shape_. the "hyper-interval" is half-open, so
    // the origin is contained within the box but the corner at
    // flattened_shape_ is not.
    return detail::is_bounded_by(idx, flattened_shape_);
  }

  template<class Result, class OuterArg, class... InnerArgs>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Predecessor& predecessor, Result& result, OuterArg& outer_arg, detail::unit, InnerArgs&... inner_args) const
  {
    flattened_index_type flattened_idx = flatten_index(idx);

    if(in_domain(flattened_idx))
    {
      f_(flattened_idx, predecessor, result, outer_arg, inner_args...);
    }
  }
};


// this specialization is for when the predecessor future's type is void
template<class Index, class Function, class Shape>
struct flatten_index_and_invoke<Index,void,Function,Shape>
{
  using index_type = Index;
  using shape_type = Shape;

  using flattened_index_type = flattened_index_type_t<Index>;
  using flattened_shape_type = flattened_shape_type_t<Shape>;

  mutable Function     f_;
  shape_type           shape_;
  flattened_shape_type flattened_shape_;

  __AGENCY_ANNOTATION
  flatten_index_and_invoke(const Function& f, shape_type shape, flattened_shape_type flattened_shape)
    : f_(f),
      shape_(shape),
      flattened_shape_(flattened_shape)
  {}

  __AGENCY_ANNOTATION
  flattened_index_type flatten_index(const Index& idx) const
  {
    return detail::merge_front_index_elements(idx, shape_);
  }

  __AGENCY_ANNOTATION
  bool in_domain(const flattened_index_type& idx) const
  {
    // idx is in the domain of f_ if idx is contained within the
    // axis-aligned bounded box from extremal corners at the origin
    // and flattened_shape_. the "hyper-interval" is half-open, so
    // the origin is contained within the box but the corner at
    // flattened_shape_ is not.
    return detail::is_bounded_by(idx, flattened_shape_);
  }

  // note that because the predecessor future type is void, no predecessor argument
  // appears in operator()'s parameter list
  template<class Result, class OuterArg, class... InnerArgs>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Result& result, OuterArg& outer_arg, detail::unit, InnerArgs&... inner_args) const
  {
    flattened_index_type flattened_idx = flatten_index(idx);

    if(in_domain(flattened_idx))
    {
      f_(flattened_idx, result, outer_arg, inner_args...);
    }
  }
};


template<class Index, class Predecessor, class Function, class Shape>
__AGENCY_ANNOTATION
flatten_index_and_invoke<Index,Predecessor,Function,Shape>
  make_flatten_index_and_invoke(Function f,
                                Shape higher_dimensional_shape,
                                typename flatten_index_and_invoke<Index,Predecessor,Function,Shape>::flattened_shape_type lower_dimensional_shape)
{
  return flatten_index_and_invoke<Index,Predecessor,Function,Shape>{f,higher_dimensional_shape,lower_dimensional_shape};
}


} // end detail


template<class Executor>
class flattened_executor
{
  // probably shouldn't insist on a scoped executor
  static_assert(
    detail::is_scoped_executor<Executor>::value,
    "Executor must have a scoped bulk guarantee."
  );

  private:
    constexpr static auto execution_depth = executor_execution_depth<Executor>::value - 1;

  public:
    using base_executor_type = Executor;
    using shape_type = detail::flattened_shape_type_t<executor_shape_t<base_executor_type>>;
    using index_type = detail::flattened_index_type_t<executor_shape_t<base_executor_type>>;

    template<class T>
    using future = executor_future_t<base_executor_type, T>;

    template<class T>
    using allocator = executor_allocator_t<base_executor_type, T>;

    using barrier_type = detail::flattened_barrier_type_t<detail::member_barrier_type_or_t<base_executor_type, void>>;

    __AGENCY_ANNOTATION
    future<void> make_ready_future() const
    {
      return agency::make_ready_future<void>(base_executor());
    }

    __AGENCY_ANNOTATION
    constexpr flattened_executor(const base_executor_type& base_executor = base_executor_type())
      : base_executor_(base_executor)
    {}

    __AGENCY_ANNOTATION
    const base_executor_type& base_executor() const
    {
      return base_executor_;
    }

    __AGENCY_ANNOTATION
    base_executor_type& base_executor()
    {
      return base_executor_;
    }

    template<__AGENCY_REQUIRES(
               detail::has_static_query<bulk_guarantee_t, base_executor_type>::value
            )>
    __AGENCY_ANNOTATION
    constexpr static auto query(const bulk_guarantee_t& prop) ->
      decltype(detail::flatten_bulk_guarantee(bulk_guarantee_t::template static_query<base_executor_type>()))
    {
      return detail::flatten_bulk_guarantee(bulk_guarantee_t::template static_query<base_executor_type>());
    }

    template<__AGENCY_REQUIRES(
               !detail::has_static_query<bulk_guarantee_t, base_executor_type>::value
            )>
    __AGENCY_ANNOTATION
    constexpr auto query(const bulk_guarantee_t& prop) const ->
      decltype(detail::flatten_bulk_guarantee(agency::query(this->base_executor(), prop)))
    {
      return detail::flatten_bulk_guarantee(agency::query(base_executor(), prop));
    }

    template<class Function, class Future, class ResultFactory, class OuterFactory, class... InnerFactories,
             __AGENCY_REQUIRES(sizeof...(InnerFactories) == execution_depth - 1)
            >
    __AGENCY_ANNOTATION
    future<detail::result_of_t<ResultFactory()>>
      bulk_then_execute(Function f, shape_type shape, Future& predecessor, ResultFactory result_factory, OuterFactory outer_factory, InnerFactories... inner_factories) const
    {
      base_shape_type base_shape = partition_into_base_shape(shape);

      using base_index_type = executor_index_t<base_executor_type>;
      using future_result_type = future_result_t<Future>;
      auto execute_me = detail::make_flatten_index_and_invoke<base_index_type,future_result_type>(f, base_shape, shape);

      return detail::bulk_then_execute(base_executor(), execute_me, base_shape, predecessor, result_factory, outer_factory, agency::detail::unit_factory(), inner_factories...);
    }

    __AGENCY_ANNOTATION
    shape_type unit_shape() const
    {
      // to flatten the base executor's shape we merge the two front dimensions together
      return detail::merge_front_shape_elements(agency::unit_shape(base_executor()));
    }

    __AGENCY_ANNOTATION
    shape_type max_shape_dimensions() const
    {
      // to flatten the base executor's shape we merge the two front dimensions together
      return detail::merge_front_shape_elements(agency::max_shape_dimensions(base_executor()));
    }

    __AGENCY_ANNOTATION
    friend bool operator==(const flattened_executor& a, const flattened_executor& b) noexcept
    {
      return a.base_executor() == b.base_executor();
    }

    __AGENCY_ANNOTATION
    friend bool operator!=(const flattened_executor& a, const flattened_executor& b) noexcept
    {
      return !(a == b);
    }

  private:
    using base_shape_type = executor_shape_t<base_executor_type>;

    static_assert(detail::is_tuple_like<base_shape_type>::value, "The shape_type of flattened_executor's base_executor must be tuple-like.");

    using shape_head_type = detail::shape_head_t<shape_type>;

    using shape_tail_type = detail::shape_tail_t<shape_type>;

    using head_partition_type = tuple<
      typename std::tuple_element<0,base_shape_type>::type,
      typename std::tuple_element<1,base_shape_type>::type
    >;

    __AGENCY_ANNOTATION
    head_partition_type partition_head(const shape_head_type& shape) const
    {
      // avoid division by zero outer_size below
      size_t requested_size = detail::shape_cast<size_t>(shape);
      if(requested_size == 0)
      {
        return head_partition_type{};
      }

      base_shape_type base_executor_shape = agency::unit_shape(base_executor());

      size_t outer_granularity = detail::index_space_size_of_shape_head(base_executor_shape);
      size_t inner_granularity = detail::index_space_size(agency::get<1>(base_executor_shape));

      base_shape_type base_executor_max_sizes = detail::max_sizes(agency::max_shape_dimensions(base_executor()));

      size_t outer_max_size = detail::shape_head(base_executor_max_sizes);
      size_t inner_max_size = agency::get<1>(base_executor_max_sizes);

      // set outer subscription to 1
      size_t outer_size = detail::min(outer_max_size, detail::min(requested_size, outer_granularity));

      size_t inner_size = (requested_size + outer_size - 1) / outer_size;

      // address inner underutilization
      // XXX consider trying to balance the utilization
      while(inner_size < inner_granularity)
      {
        // halve the outer size
        outer_size = detail::max<int>(1, outer_size / 2);
        inner_size *= 2;
      }

      // we may require one partially-full group of agents
      if(outer_size * inner_size < requested_size)
      {
        // we require a single partially-full group of agents
        outer_size += 1;
      }

      if(inner_size > inner_max_size)
      {
        inner_size = inner_max_size;
        outer_size = (requested_size + inner_size - 1) / inner_size;

        if(outer_size > outer_max_size)
        {
          // XXX we could have asserted size <= outer_max_size * inner_max_size at the very beginning
#ifndef __CUDA_ARCH__
          throw std::runtime_error("flattened_executor::partition_head(): size is too large to accomodate");
#else
          printf("flattened_executor::partition_head(): size is too large to accomodate\n");
          assert(0);
#endif
        }
      }

      assert(outer_size * inner_size >= requested_size);

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
    __AGENCY_ANNOTATION
    static base_shape_type make_base_shape_impl(detail::index_sequence<Indices...>, const head_partition_type& partition_of_head, const shape_tail_type& tail)
    {
      return base_shape_type{agency::get<0>(partition_of_head), agency::get<1>(partition_of_head), agency::get<Indices>(tail)...};
    }

    __AGENCY_ANNOTATION
    static base_shape_type make_base_shape(const head_partition_type& partition_of_head, const shape_tail_type& tail)
    {
      auto indices = detail::make_index_sequence<std::tuple_size<shape_tail_type>::value>();

      return make_base_shape_impl(indices, partition_of_head, tail);
    }

    __AGENCY_ANNOTATION
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

