#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/executor/new_executor_traits.hpp>
#include <agency/execution/executor/customization_points/bulk_then_execute.hpp>
#include <agency/detail/invoke.hpp>

namespace agency
{
namespace detail
{


// this adaptor turns an Executor into a BulkContinuationExecutor
// XXX eliminate this when Agency drops support for legacy executors 
template<class E, bool Enable = BulkExecutor<E>()>
class bulk_continuation_executor_adaptor;

template<class BulkExecutor>
class bulk_continuation_executor_adaptor<BulkExecutor,true>
{
  private:
    BulkExecutor adapted_executor_;

  public:
    using execution_category = member_execution_category_or_t<BulkExecutor, unsequenced_execution_tag>;
    using shape_type = new_executor_shape_t<BulkExecutor>;
    using index_type = new_executor_index_t<BulkExecutor>;

    template<class T>
    using future = executor_future_t<BulkExecutor,T>;

    template<class T>
    using allocator = new_executor_allocator_t<BulkExecutor,T>;

    __AGENCY_ANNOTATION
    bulk_continuation_executor_adaptor() = default;

    __AGENCY_ANNOTATION
    bulk_continuation_executor_adaptor(const bulk_continuation_executor_adaptor&) = default;

    __AGENCY_ANNOTATION
    bulk_continuation_executor_adaptor(const BulkExecutor& other)
      : adapted_executor_(other)
    {}

    template<class Function, class Future, class ResultFactory, class... SharedFactories>
    __AGENCY_ANNOTATION
    future<result_of_t<ResultFactory()>>
      bulk_then_execute(Function f, shape_type shape, Future& predecessor, ResultFactory result_factory, SharedFactories... shared_factories)
    {
      return agency::bulk_then_execute(adapted_executor_, f, shape, predecessor, result_factory, shared_factories...);
    }
};


template<class LegacyExecutor>
class bulk_continuation_executor_adaptor<LegacyExecutor,false>
{
  private:
    LegacyExecutor adapted_executor_;

  public:
    using execution_category = typename executor_traits<LegacyExecutor>::execution_category;
    using shape_type = typename executor_traits<LegacyExecutor>::shape_type;
    using index_type = typename executor_traits<LegacyExecutor>::index_type;

    template<class T>
    using future = typename executor_traits<LegacyExecutor>::template future<T>;

    template<class T>
    using allocator = typename executor_traits<LegacyExecutor>::template allocator<T>;

    __AGENCY_ANNOTATION
    bulk_continuation_executor_adaptor() = default;

    __AGENCY_ANNOTATION
    bulk_continuation_executor_adaptor(const bulk_continuation_executor_adaptor&) = default;

    __AGENCY_ANNOTATION
    bulk_continuation_executor_adaptor(const LegacyExecutor& other)
      : adapted_executor_(other)
    {}

    template<class ResultFactory>
    struct legacy_result_factory_adaptor
    {
      mutable ResultFactory f;
    
      template<class Shape>
      __AGENCY_ANNOTATION
      auto operator()(const Shape&) const
        -> decltype(agency::detail::invoke(f))
      {
        return agency::detail::invoke(f);
      }
    };

    template<class Function, class Future, class ResultFactory, class... SharedFactories>
    __AGENCY_ANNOTATION
    future<result_of_t<ResultFactory()>>
      bulk_then_execute(Function f, shape_type shape, Future& predecessor, ResultFactory result_factory, SharedFactories... shared_factories)
    {
      using result_type = result_of_t<ResultFactory()>;

      future<result_type> results = executor_traits<LegacyExecutor>::template make_ready_future<result_type>(adapted_executor_, result_factory());

      auto futures = detail::make_tuple(std::move(predecessor), std::move(results));

      return executor_traits<LegacyExecutor>::template when_all_execute_and_select<1>(adapted_executor_, f, shape, futures, shared_factories...);
    }
};


} // end experimental
} // end agency

