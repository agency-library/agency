#pragma once

#include <utility>
#include <agency/execution_categories.hpp>
#include <agency/executor_traits.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/functional.hpp>
#include <agency/detail/index_tuple.hpp>
#include <agency/detail/shape_tuple.hpp>
#include <agency/detail/tuple_utility.hpp>
#include <agency/detail/unwrap_tuple_if_not_nested.hpp>

namespace agency
{


template<class Executor1, class Executor2>
class nested_executor
{
  public:
    using outer_executor_type = Executor1;
    using inner_executor_type = Executor2;

  private:
    using outer_traits = executor_traits<outer_executor_type>;
    using inner_traits = executor_traits<inner_executor_type>;

    using outer_execution_category = typename outer_traits::execution_category;
    using inner_execution_category = typename inner_traits::execution_category;

    using outer_index_type = typename outer_traits::index_type;
    using inner_index_type = typename inner_traits::index_type;

  public:
    using index_type = detail::nested_index_t<
      outer_execution_category,
      inner_execution_category,
      outer_index_type,
      inner_index_type
    >;

  private:
    using outer_shape_type = typename outer_traits::shape_type;
    using inner_shape_type = typename inner_traits::shape_type;

    static index_type make_index(const outer_index_type& outer_idx, const inner_index_type& inner_idx)
    {
      return detail::make_nested_index<outer_execution_category,inner_execution_category>(outer_idx, inner_idx);
    }

  public:
    using execution_category = 
      nested_execution_tag<
        outer_execution_category,
        inner_execution_category
      >;

    using shape_type = detail::nested_shape_t<
      outer_execution_category,
      inner_execution_category,
      outer_shape_type,
      inner_shape_type
    >;

    template<class T>
    using future = typename outer_traits::template future<T>;

    template<class T>
    using container = typename outer_traits::template container<T>;

    future<void> make_ready_future()
    {
      return outer_traits::make_ready_future(outer_executor());
    }

    nested_executor() = default;

    nested_executor(const outer_executor_type& outer_ex,
                    const inner_executor_type& inner_ex)
      : outer_ex_(outer_ex),
        inner_ex_(inner_ex)
    {}

    // XXX executor adaptors like nested_executor may need to implement the entire interface to be sure we get the most efficient implementation

  private:
    // this is the functor used by then_execute below
    // it takes the place of a nested, polymorphic lambda function
    template<class Function, class... InnerFactories>
    struct then_execute_outer_functor
    {
      nested_executor&                 exec;
      Function                         f;
      inner_shape_type                 inner_shape;
      detail::tuple<InnerFactories...> inner_factories;

      template<class OuterIndex, class OuterSharedType>
      struct then_execute_inner_functor
      {
        Function f;
        OuterIndex outer_idx;
        OuterSharedType& outer_shared;

        template<class InnerIndex, class... T>
        void operator()(const InnerIndex& inner_idx, T&... inner_shared)
        {
          f(make_index(outer_idx, inner_idx), outer_shared, inner_shared...);
        }
      };

      template<class OuterIndex, class PastSharedType, class OuterSharedType>
      struct then_execute_inner_functor_with_past_parameter
      {
        Function f;
        OuterIndex outer_idx;
        PastSharedType& past_shared;
        OuterSharedType& outer_shared;

        template<class InnerIndex, class... T>
        void operator()(const InnerIndex& inner_idx, T&... inner_shared)
        {
          f(make_index(outer_idx, inner_idx), past_shared, outer_shared, inner_shared...);
        }
      };

      template<size_t... Indices, class OuterIndex, class T>
      void invoke_execute(detail::index_sequence<Indices...>, const OuterIndex& outer_idx, T& outer_shared)
      {
        inner_traits::execute(exec.inner_executor(), then_execute_inner_functor<OuterIndex,T>{f, outer_idx, outer_shared}, inner_shape, detail::get<Indices>(inner_factories)...);
      }

      template<size_t... Indices, class OuterIndex, class T1, class T2>
      void invoke_execute(detail::index_sequence<Indices...>, const OuterIndex& outer_idx, T1& past_shared, T2& outer_shared)
      {
        inner_traits::execute(exec.inner_executor(), then_execute_inner_functor_with_past_parameter<OuterIndex,T1,T2>{f, outer_idx, past_shared, outer_shared}, inner_shape, detail::get<Indices>(inner_factories)...);
      }

      template<class OuterIndex, class T>
      void operator()(const OuterIndex& outer_idx, T& outer_shared)
      {
        invoke_execute(detail::index_sequence_for<InnerFactories...>(), outer_idx, outer_shared);
      }

      template<class OuterIndex, class T1, class T2>
      void operator()(const OuterIndex& outer_idx, T1& past_shared, T2& outer_shared)
      {
        invoke_execute(detail::index_sequence_for<InnerFactories...>(), outer_idx, past_shared, outer_shared);
      }
    };
    

  public:
    // XXX Future is a template parameter because future<T> is an alias, which interferes with template deduction
    template<class Function, class Future, class Factory, class... Factories>
    future<void> then_execute(Function f, shape_type shape, Future& fut, Factory outer_factory, Factories... inner_factories)
    {
      static_assert(detail::execution_depth<execution_category>::value == 1 + sizeof...(Factories), "Number of factories must be the same as the depth of execution_category.");

      // split the shape into the inner & outer portions
      auto outer_shape = this->outer_shape(shape);
      auto inner_shape = this->inner_shape(shape);

      return outer_traits::then_execute(
        outer_executor(),
        then_execute_outer_functor<Function,Factories...>{*this, f, inner_shape, detail::make_tuple(inner_factories...)},
        outer_shape,
        fut,
        outer_factory
      );

      // XXX use this implementation upon c++14:
      //return outer_traits::then_execute(outer_executor(), [=](const auto& outer_idx, auto& past_shared_param, auto& outer_shared_param)
      //{
      //  inner_traits::execute(inner_executor(), [=,&outer_shared_param](const auto& inner_idx, auto&... inner_shared_parms)
      //  {
      //    f(make_index(outer_idx, inner_idx), past_shared_param, outer_shared_param, inner_shared_params...);
      //  },
      //  inner_shape,
      //  inner_factories...
      //  );
      //},
      //outer_shape,
      //fut,
      //outer_factory
      //);
    }

    outer_executor_type& outer_executor()
    {
      return outer_ex_;
    }

    const outer_executor_type& outer_executor() const
    {
      return outer_ex_;
    }

    inner_executor_type& inner_executor()
    {
      return inner_ex_;
    }

    const inner_executor_type& inner_executor() const
    {
      return inner_ex_;
    }

    shape_type shape() const
    {
      auto outer_exec_shape = outer_traits::shape(outer_executor());
      auto inner_exec_shape = inner_traits::shape(inner_executor());

      return make_shape(outer_exec_shape, inner_exec_shape);
    }

  private:
    static outer_shape_type outer_shape(const shape_type& shape)
    {
      // the outer portion is always the head of the tuple
      return __tu::tuple_head(shape);
    }

    static inner_shape_type inner_shape(const shape_type& shape)
    {
      // the inner portion is the tail of the tuple, but if the 
      // inner executor is not nested, then the tuple needs to be unwrapped
      return detail::unwrap_tuple_if_not_nested<inner_execution_category>(detail::forward_tail(shape));
    }

    __AGENCY_ANNOTATION
    static shape_type make_shape(const outer_shape_type& outer_shape, const inner_shape_type& inner_shape)
    {
      return detail::make_nested_shape<outer_execution_category,inner_execution_category>(outer_shape, inner_shape);
    }

    outer_executor_type outer_ex_;
    inner_executor_type inner_ex_;
};


} // end agency

