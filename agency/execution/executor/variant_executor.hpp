#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/experimental/variant.hpp>
#include <agency/future/variant_future.hpp>
#include <agency/future/detail/future_sum.hpp>
#include <agency/coordinate/detail/shape/common_shape.hpp>
#include <agency/memory/allocator/detail/allocator_sum.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/executor/customization_points.hpp>
#include <agency/execution/executor/detail/adaptors/executor_ref.hpp>
#include <agency/execution/executor/properties.hpp>
#include <agency/execution/executor/require.hpp>
#include <agency/execution/executor/properties/detail/common_bulk_guarantee.hpp>
#include <agency/execution/executor/properties/detail/bulk_guarantee_depth.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/tuple.hpp>

#include <type_traits>
#include <utility>
#include <typeinfo>

namespace agency
{


template<class Executor, class... Executors>
class variant_executor
{
  private:
    using variant_type = experimental::variant<Executor,Executors...>;

    using bulk_guarantee_type = detail::common_bulk_guarantee_t<
      decltype(bulk_guarantee_t::static_query<Executor>()),
      decltype(bulk_guarantee_t::static_query<Executors>())...
    >;

  public:
    /// variant_executor's bulk_guarantee_t provides the strongest guarantee
    /// permitted by the union of the guarantees provided by the alternative executors
    /// When no static guarantee may be provided, the guarantee is unsequenced_t.
    __AGENCY_ANNOTATION
    constexpr static bulk_guarantee_type query(bulk_guarantee_t)
    {
      return bulk_guarantee_type();
    }

  private:
    static constexpr size_t execution_depth = detail::bulk_guarantee_depth<bulk_guarantee_type>::value;

  public:
    /// variant_executor's associated future type is sum type for futures.
    //  This type is a variant_future whose alternatives are taken from the list of the alternative executors' futures.
    /// Only the unique futures in this list are alternatives of this variant_future.
    /// If there is only a single unique future type, then variant_executor's associated future is simply that unique future.
    template<class T>
    using future = detail::future_sum_t<
      executor_future_t<Executor,T>,
      executor_future_t<Executors,T>...
    >;

    using shape_type = detail::common_shape_t<
      executor_shape_t<Executor>,
      executor_shape_t<Executors>...
    >;

    /// variant_executor's associated allocator is a sum type for allocators.
    /// This type is a variant_allocator whose alternatives are taken from the list of the alternative executors' allocators.
    /// Only the unique allocators in this list are alternative of this variant_allocator.
    /// If there is only a single unique allocator type, then variant_allocator's associated allocator is simply that unique allocator.
    template<class T>
    using allocator = detail::allocator_sum_t<
      executor_allocator_t<Executor,T>,
      executor_allocator_t<Executors,T>...
    >;

    variant_executor() = default;

    variant_executor(const variant_executor& other) = default;

    template<class OtherExecutor,
             __AGENCY_REQUIRES(
               std::is_constructible<variant_type, OtherExecutor&&>::value
             )>
    __AGENCY_ANNOTATION
    variant_executor(OtherExecutor&& other)
      : variant_(std::forward<OtherExecutor>(other))
    {}

  private:
    struct type_visitor
    {
      template<class E>
      const std::type_info& operator()(const E& exec) const
      {
        return typeid(exec);
      }
    };

  public:
    const std::type_info& type() const
    {
      return experimental::visit(type_visitor(), variant_);
    }

    __AGENCY_ANNOTATION
    std::size_t index() const
    {
      return variant_.index();
    }

    // customization points follow
    //
    // the implementation of each follows the same pattern:
    // 1. define one (possibly two) visitor types that visit an alternative executor
    //    and call the corresponding customization point
    // 2. the member function creates a visitor of the appropriate type and calls experimental::visit
    //
    // There's nothing particularly interesting happening -- the goal of each member function is simply
    // to forward its parameters to the active alternative via visitation.
    // 
    // Two notes:
    //
    // 1. Our shape_type must be converted to the alternative's shape_type inside the visitor via shape_cast()
    //
    // 2. Functions that take a future as a parameter have specializations for foreign Futures
    //    and variant_futures. When a variant_future is encountered, the visitor visits both the variant_executor and variant_future
    //    simultaneously.

    // twoway_execute
  private:
    template<class FunctionRef>
    struct twoway_execute_visitor
    {
      FunctionRef f;

      __agency_exec_check_disable__
      template<class E>
      __AGENCY_ANNOTATION
      future<detail::result_of_t<detail::decay_t<FunctionRef>()>>
        operator()(E& exec) const
      {
        detail::executor_ref<E> exec_ref{exec};
        return agency::require(exec_ref, single, twoway).twoway_execute(std::forward<FunctionRef>(f));
      }
    };

  public:
    template<class Function>
    __AGENCY_ANNOTATION
    future<detail::result_of_t<detail::decay_t<Function>()>>
    twoway_execute(Function&& f) const
    {
      return experimental::visit(twoway_execute_visitor<Function&&>{std::forward<Function>(f)}, variant_);
    }
    
    // bulk_twoway_execute
  private:
    template<class Function, class ResultFactory, class... SharedFactories>
    struct bulk_twoway_execute_visitor
    {
      Function f;
      shape_type shape;
      ResultFactory result_factory;
      tuple<SharedFactories...> shared_factories;

      __agency_exec_check_disable__
      template<class E, size_t... Indices>
      __AGENCY_ANNOTATION
      future<detail::result_of_t<ResultFactory()>>
        impl(detail::index_sequence<Indices...>, const E& exec) const
      {
        // cast from our shape type to E's shape type
        executor_shape_t<E> casted_shape = detail::shape_cast<executor_shape_t<E>>(shape);

        detail::executor_ref<E> exec_ref{exec};
        return agency::require(exec_ref, bulk, twoway).bulk_twoway_execute(f, casted_shape, result_factory, agency::get<Indices>(shared_factories)...);
      }

      __agency_exec_check_disable__
      template<class E>
      __AGENCY_ANNOTATION
      future<detail::result_of_t<ResultFactory()>>
        operator()(const E& exec) const
      {
        return impl(detail::make_index_sequence<sizeof...(SharedFactories)>(), exec);
      }
    };

  public:
    template<class Function, class ResultFactory, class... SharedFactories,
             __AGENCY_REQUIRES(execution_depth == sizeof...(SharedFactories))
            >
    __AGENCY_ANNOTATION
    future<detail::result_of_t<ResultFactory()>>
    bulk_twoway_execute(Function f, shape_type shape, ResultFactory result_factory, SharedFactories... shared_factories) const
    {
      auto visitor = bulk_twoway_execute_visitor<Function,ResultFactory,SharedFactories...>{f, shape, result_factory, agency::make_tuple(shared_factories...)};
      return experimental::visit(visitor, variant_);
    }

    // bulk_then_execute
  private:
    // this is a unary visitor that only visits variant_executor
    template<class Function, class Future, class ResultFactory, class... SharedFactories>
    struct bulk_then_execute_visitor1
    {
      Function f;
      shape_type shape;
      Future& predecessor_future;
      ResultFactory result_factory;
      tuple<SharedFactories...> shared_factories;

      template<class E, size_t... Indices>
      __AGENCY_ANNOTATION
      future<detail::result_of_t<ResultFactory()>>
        impl(detail::index_sequence<Indices...>, E& exec) const
      {
        // cast from our shape type to E's shape type
        executor_shape_t<E> casted_shape = detail::shape_cast<executor_shape_t<E>>(shape);

        detail::executor_ref<E> exec_ref{exec};
        return agency::require(exec_ref, agency::bulk, agency::then).bulk_then_execute(f, casted_shape, predecessor_future, result_factory, agency::get<Indices>(shared_factories)...);
      }

      template<class E>
      __AGENCY_ANNOTATION
      future<detail::result_of_t<ResultFactory()>>
        operator()(E& exec) const
      {
        return impl(detail::make_index_sequence<sizeof...(SharedFactories)>(), exec);
      }
    };

    // this is a binary visitor that visits variant_executor & variant_future simultaneously
    template<class Function, class ResultFactory, class... SharedFactories>
    struct bulk_then_execute_visitor2
    {
      Function f;
      shape_type shape;
      ResultFactory result_factory;
      tuple<SharedFactories...> shared_factories;

      __agency_exec_check_disable__
      template<class E, class VariantFuture, size_t... Indices>
      __AGENCY_ANNOTATION
      future<detail::result_of_t<ResultFactory()>>
        impl(detail::index_sequence<Indices...>, E& exec, VariantFuture& predecessor_future) const
      {
        // cast from our shape type to E's shape type
        executor_shape_t<E> casted_shape = detail::shape_cast<executor_shape_t<E>>(shape);

        detail::executor_ref<E> exec_ref{exec};

        return agency::require(exec_ref, bulk, then).bulk_then_execute(f, casted_shape, predecessor_future, result_factory, agency::get<Indices>(shared_factories)...);
      }

      __agency_exec_check_disable__
      template<class E, class Future>
      __AGENCY_ANNOTATION
      future<detail::result_of_t<ResultFactory()>>
        operator()(E& exec, Future& predecessor_future) const
      {
        return impl(detail::make_index_sequence<sizeof...(SharedFactories)>(), exec, predecessor_future);
      }
    };

  public:
    // this overload of bulk_then_execute() is for the case when Future is an instance of variant_future
    template<class Function, class VariantFuture, class ResultFactory, class... SharedFactories,
             __AGENCY_REQUIRES(execution_depth == sizeof...(SharedFactories)),
             __AGENCY_REQUIRES(detail::is_variant_future<VariantFuture>::value)
            >
    __AGENCY_ANNOTATION
    future<detail::result_of_t<ResultFactory()>>
    bulk_then_execute(Function f, shape_type shape,
                      VariantFuture& predecessor_future,
                      ResultFactory result_factory,
                      SharedFactories... shared_factories) const
    {
      auto visitor = bulk_then_execute_visitor2<Function,ResultFactory,SharedFactories...>{f, shape, result_factory, agency::make_tuple(shared_factories...)};
      auto future_variant = predecessor_future.variant();
      return experimental::visit(visitor, variant_, future_variant);
    }

    // this overload of bulk_then_execute() is for the case when Future is not an instance of variant_future
    template<class Function, class Future, class ResultFactory, class... SharedFactories,
             __AGENCY_REQUIRES(execution_depth == sizeof...(SharedFactories)),
             __AGENCY_REQUIRES(!detail::is_variant_future<Future>::value)
            >
    __AGENCY_ANNOTATION
    future<detail::result_of_t<ResultFactory()>>
    bulk_then_execute(Function f, shape_type shape,
                      Future& predecessor_future,
                      ResultFactory result_factory,
                      SharedFactories... shared_factories) const
    {
      auto visitor = bulk_then_execute_visitor1<Function,Future,ResultFactory,SharedFactories...>{f, shape, predecessor_future, result_factory, agency::make_tuple(shared_factories...)};
      return experimental::visit(visitor, variant_);
    }

  private:
    // this is a unary visitor that only visits variant_executor
    template<class T, class VariantFuture>
    struct future_cast_visitor1
    {
      VariantFuture& fut;

      template<class E>
      __AGENCY_ANNOTATION
      future<T> operator()(const E& exec) const
      {
        return agency::future_cast<T>(exec, fut);
      }
    };

    // this is a binary visitor that visits variant_executor & variant_future simultaneously
    template<class T>
    struct future_cast_visitor2
    {
      template<class E, class Future>
      __AGENCY_ANNOTATION
      future<T> operator()(const E& exec, Future& fut) const
      {
        return agency::future_cast<T>(exec, fut);
      }
    };


  public:
    // this overload of future_cast() is for the case when the future to cast is an instance of variant_future
    template<class T, class VariantFuture,
             __AGENCY_REQUIRES(detail::is_variant_future<VariantFuture>::value)
            >
    __AGENCY_ANNOTATION
    future<T> future_cast(VariantFuture& fut) const
    {
      auto visitor = future_cast_visitor2<T>();
      auto future_variant = fut.variant();
      return experimental::visit(visitor, variant_, future_variant);
    }

    // this overload of future_cast() is for the case when the future to cast is not an instance of variant_future
    template<class T, class Future,
             __AGENCY_REQUIRES(!detail::is_variant_future<Future>::value)
            >
    __AGENCY_ANNOTATION
    future<T> future_cast(Future& fut) const
    {
      auto visitor = future_cast_visitor1<T,Future>{fut};
      return experimental::visit(visitor, variant_);
    }

    // make_ready_future
  private:
    template<class T, class... Args>
    struct make_ready_future_visitor
    {
      tuple<Args...> args;

      __agency_exec_check_disable__
      template<class E, size_t... Indices>
      __AGENCY_ANNOTATION
      future<T> impl(detail::index_sequence<Indices...>, const E& exec) const
      {
        return agency::make_ready_future<T>(exec, agency::get<Indices>(args)...);
      }

      __agency_exec_check_disable__
      template<class E>
      __AGENCY_ANNOTATION
      future<T> operator()(const E& exec) const
      {
        return impl(detail::make_index_sequence<sizeof...(Args)>(), exec);
      }
    };

  public:
    template<class T, class... Args>
    __AGENCY_ANNOTATION
    future<T> make_ready_future(Args&&... args) const
    {
      auto args_tuple = agency::forward_as_tuple(std::forward<Args>(args)...);
      auto visitor = make_ready_future_visitor<T,Args&&...>{args_tuple};
      return experimental::visit(visitor, variant_);
    }

    // max_shape_dimensions
  private:
    struct max_shape_dimensions_visitor
    {
      template<class E>
      __AGENCY_ANNOTATION
      shape_type operator()(const E& exec) const
      {
        return detail::shape_cast<shape_type>(agency::max_shape_dimensions(exec));
      }
    };

  public:
    __AGENCY_ANNOTATION
    shape_type max_shape_dimensions() const
    {
      return experimental::visit(max_shape_dimensions_visitor(), variant_);
    }
    
    // then_execute
  private:
    // this is a unary visitor that only visits variant_executor
    template<class FunctionRef, class Future>
    struct then_execute_visitor1
    {
      FunctionRef f;
      Future& predecessor_future;

      template<class E>
      __AGENCY_ANNOTATION
      future<detail::result_of_continuation_t<detail::decay_t<FunctionRef>, Future>>
        operator()(const E& exec) const
      {
        return detail::then_execute(exec, std::forward<FunctionRef>(f), predecessor_future);
      }
    };

    // this is a binary visitor that visits variant_executor & variant_future simultaneously
    template<class FunctionRef>
    struct then_execute_visitor2
    {
      FunctionRef f;

      template<class E, class Future>
      __AGENCY_ANNOTATION
      future<detail::result_of_continuation_t<detail::decay_t<FunctionRef>, Future>>
        operator()(const E& exec, Future& predecessor_future) const
      {
        detail::executor_ref<E> exec_ref{exec};
        return agency::require(exec_ref, agency::single, agency::then).then_execute(std::forward<FunctionRef>(f), predecessor_future);
      }
    };

  public:
    // this overload of then_execute() is for the case when the predecessor future is an instance of variant_future
    template<class Function, class VariantFuture,
             __AGENCY_REQUIRES(detail::is_variant_future<VariantFuture>::value)
            >
    __AGENCY_ANNOTATION
    future<detail::result_of_continuation_t<detail::decay_t<Function>, VariantFuture>>
    then_execute(Function&& f, VariantFuture& predecessor_future) const
    {
      auto visitor = then_execute_visitor2<Function&&>{std::forward<Function>(f)};
      auto future_variant = predecessor_future.variant();
      return experimental::visit(visitor, variant_, future_variant);
    }

    // this overload of then_execute() is for the case when the predecessor future is not an instance of variant_future
    template<class Function, class Future,
             __AGENCY_REQUIRES(!detail::is_variant_future<Future>::value)
            >
    __AGENCY_ANNOTATION
    future<detail::result_of_continuation_t<detail::decay_t<Function>, Future>>
    then_execute(Function&& f, Future& predecessor_future) const
    {
      auto visitor = then_execute_visitor1<Function&&,Future>{std::forward<Function>(f), predecessor_future};
      return experimental::visit(visitor, variant_);
    }

    // unit_shape
  private:
    struct unit_shape_visitor
    {
      template<class E>
      __AGENCY_ANNOTATION
      shape_type operator()(const E& exec) const
      {
        auto result = agency::unit_shape(exec);
        return detail::shape_cast<shape_type>(result);
      }
    };

  public:
    __AGENCY_ANNOTATION
    shape_type unit_shape() const
    {
      return experimental::visit(unit_shape_visitor(), variant_);
    }

  private:
    variant_type variant_;
};


} // end agency

