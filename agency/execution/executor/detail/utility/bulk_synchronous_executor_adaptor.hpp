#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/executor/customization_points/bulk_sync_execute.hpp>
#include <agency/detail/invoke.hpp>


namespace agency
{
namespace detail
{


// this adaptor turns an Executor into a BulkSynchronousExecutor
// XXX nomerge
// XXX eliminate this when Agency drops support for legacy executors 
template<class E, bool Enable = is_executor<E>::value>
class bulk_synchronous_executor_adaptor;

template<class Executor>
class bulk_synchronous_executor_adaptor<Executor,true>
{
  private:
    Executor adapted_executor_;

  public:
    using execution_category = member_execution_category_or_t<Executor, unsequenced_execution_tag>;
    using shape_type = executor_shape_t<Executor>;
    using index_type = executor_index_t<Executor>;

    template<class T>
    using allocator = executor_allocator_t<Executor,T>;

    __AGENCY_ANNOTATION
    bulk_synchronous_executor_adaptor() = default;

    __AGENCY_ANNOTATION
    bulk_synchronous_executor_adaptor(const bulk_synchronous_executor_adaptor&) = default;

    __AGENCY_ANNOTATION
    bulk_synchronous_executor_adaptor(const Executor& other)
      : adapted_executor_(other)
    {}

    template<class Function, class ResultFactory, class... SharedFactories>
    __AGENCY_ANNOTATION
    result_of_t<ResultFactory()>
      bulk_sync_execute(Function f, shape_type shape, ResultFactory result_factory, SharedFactories... shared_factories) const
    {
      return agency::bulk_sync_execute(adapted_executor_, f, shape, result_factory, shared_factories...);
    }
};


} // end detail
} // end agency

