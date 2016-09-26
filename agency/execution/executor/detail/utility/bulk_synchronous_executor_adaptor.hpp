#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/executor/new_executor_traits.hpp>
#include <agency/execution/executor/customization_points/bulk_execute.hpp>
#include <agency/detail/invoke.hpp>


namespace agency
{
namespace detail
{


// this adaptor turns an Executor into a BulkSynchronousExecutor
// XXX eliminate this when Agency drops support for legacy executors 
template<class E, bool Enable = BulkExecutor<E>()>
class bulk_synchronous_executor_adaptor;

template<class BulkExecutor>
class bulk_synchronous_executor_adaptor<BulkExecutor,true>
{
  private:
    BulkExecutor adapted_executor_;

  public:
    using execution_category = member_execution_category_or_t<BulkExecutor, unsequenced_execution_tag>;
    using shape_type = new_executor_shape_t<BulkExecutor>;
    using index_type = new_executor_index_t<BulkExecutor>;

    template<class T>
    using allocator = new_executor_allocator_t<BulkExecutor,T>;

    __AGENCY_ANNOTATION
    bulk_synchronous_executor_adaptor() = default;

    __AGENCY_ANNOTATION
    bulk_synchronous_executor_adaptor(const bulk_synchronous_executor_adaptor&) = default;

    __AGENCY_ANNOTATION
    bulk_synchronous_executor_adaptor(const BulkExecutor& other)
      : adapted_executor_(other)
    {}

    template<class Function, class ResultFactory, class... SharedFactories>
    __AGENCY_ANNOTATION
    result_of_t<ResultFactory()>
      bulk_execute(Function f, shape_type shape, ResultFactory result_factory, SharedFactories... shared_factories)
    {
      return agency::bulk_execute(adapted_executor_, f, shape, result_factory, shared_factories...);
    }
};


template<class LegacyExecutor>
class bulk_synchronous_executor_adaptor<LegacyExecutor,false>
{
  private:
    LegacyExecutor adapted_executor_;

  public:
    using execution_category = typename executor_traits<LegacyExecutor>::execution_category;
    using shape_type = typename executor_traits<LegacyExecutor>::shape_type;
    using index_type = typename executor_traits<LegacyExecutor>::index_type;

    template<class T>
    using allocator = typename executor_traits<LegacyExecutor>::template allocator<T>;

    __AGENCY_ANNOTATION
    bulk_synchronous_executor_adaptor() = default;

    __AGENCY_ANNOTATION
    bulk_synchronous_executor_adaptor(const bulk_synchronous_executor_adaptor&) = default;

    __AGENCY_ANNOTATION
    bulk_synchronous_executor_adaptor(const LegacyExecutor& other)
      : adapted_executor_(other)
    {}

    template<class Function, class Result>
    struct bulk_execute_functor
    {
      mutable Function f;
      Result& result;

      template<class Index, class... SharedArgs>
      __AGENCY_ANNOTATION
      void operator()(const Index& idx, SharedArgs&... shared_args) const
      {
        f(idx, result, shared_args...);
      }
    };

    template<class Function, class ResultFactory, class... SharedFactories>
    __AGENCY_ANNOTATION
    result_of_t<ResultFactory()>
      bulk_execute(Function f, shape_type shape, ResultFactory result_factory, SharedFactories... shared_factories)
    {
      auto result = ResultFactory();

      executor_traits<LegacyExecutor>::execute(adapted_executor_, bulk_execute_functor<Function,decltype(result)>{f,result}, shape, shared_factories...);

      return std::move(result);
    }
};


} // end detail
} // end agency
