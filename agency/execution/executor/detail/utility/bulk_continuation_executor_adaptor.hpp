#pragma once

#include <agency/detail/config.hpp>
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
    using shape_type = executor_shape_t<BulkExecutor>;
    using index_type = executor_index_t<BulkExecutor>;

    template<class T>
    using future = executor_future_t<BulkExecutor,T>;

    template<class T>
    using allocator = executor_allocator_t<BulkExecutor,T>;

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


} // end experimental
} // end agency

