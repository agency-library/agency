#pragma once

#include <utility>
#include <agency/execution/executor/executor_array.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_continuation_executor_adaptor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_synchronous_executor_adaptor.hpp>

namespace agency
{


template<class Executor1, class Executor2>
class scoped_executor : public executor_array<Executor2,Executor1>
{
  private:
    using super_t = executor_array<Executor2,Executor1>;

  public:
    using outer_executor_type = Executor1;
    using inner_executor_type = Executor2;

    scoped_executor(const outer_executor_type&,
                    const inner_executor_type& inner_ex)
      : super_t(1, inner_ex)
    {}

    scoped_executor() :
      scoped_executor(outer_executor_type(), inner_executor_type())
    {}

    // XXX eliminate all this when executor_array::bulk_then_execute() exists
    template<class T>
    using future = typename executor_traits<super_t>::template future<T>;

    using shape_type = typename executor_traits<super_t>::shape_type;


    template<class Function, class InnerFactory>
    struct bulk_then_execute_functor
    {
      using outer_index_type = typename super_t::outer_index_type;
      using inner_index_type = typename super_t::inner_index_type;
      using inner_shape_type = typename super_t::inner_shape_type;

      inner_executor_type inner_exec;
      mutable Function f;
      inner_shape_type inner_shape;
      InnerFactory inner_factory;

      template<class Predecessor, class Result, class OuterArg>
      struct inner_functor_with_predecessor
      {
        mutable Function f;
        outer_index_type outer_idx;
        Predecessor& predecessor;
        Result& result;
        OuterArg& outer_arg;
        
        template<class InnerArg>
        __AGENCY_ANNOTATION
        void operator()(inner_index_type inner_idx, detail::unit, InnerArg& inner_arg) const
        {
          auto idx = super_t::make_index(outer_idx, inner_idx);
          detail::invoke(f, idx, predecessor, result, outer_arg, inner_arg);
        }
      };

      template<class Predecessor, class Result, class OuterArg>
      __AGENCY_ANNOTATION
      void operator()(outer_index_type outer_idx, Predecessor& predecessor, Result& result, OuterArg& outer_arg) const
      {
        detail::new_executor_traits_detail::bulk_synchronous_executor_adaptor<inner_executor_type> adapted_inner_exec(inner_exec);
        adapted_inner_exec.bulk_execute(inner_functor_with_predecessor<Predecessor,Result,OuterArg>{f, outer_idx, predecessor, result, outer_arg}, inner_shape, detail::unit_factory(), inner_factory);
      }

      template<class Result, class OuterArg>
      struct inner_functor_without_predecessor
      {
        mutable Function f;
        outer_index_type outer_idx;
        Result& result;
        OuterArg& outer_arg;
        
        template<class InnerArg>
        __AGENCY_ANNOTATION
        void operator()(inner_index_type inner_idx, detail::unit, InnerArg& inner_arg) const
        {
          auto idx = super_t::make_index(outer_idx, inner_idx);
          detail::invoke(f, idx, result, outer_arg, inner_arg);
        }
      };

      template<class Result, class OuterArg>
      __AGENCY_ANNOTATION
      void operator()(outer_index_type outer_idx, Result& result, OuterArg& outer_arg) const
      {
        detail::new_executor_traits_detail::bulk_synchronous_executor_adaptor<inner_executor_type> adapted_inner_exec(inner_exec);
        adapted_inner_exec.bulk_execute(inner_functor_without_predecessor<Result,OuterArg>{f, outer_idx, result, outer_arg}, inner_shape, detail::unit_factory(), inner_factory);
      }
    };

    template<class Function, class Future, class ResultFactory, class OuterFactory, class InnerFactory>
    __AGENCY_ANNOTATION
    future<detail::result_of_t<ResultFactory()>>
    bulk_then_execute(Function f, shape_type shape, Future& predecessor, ResultFactory result_factory, OuterFactory outer_factory, InnerFactory inner_factory)
    {
      auto outer_shape = this->outer_shape(shape);
      auto inner_shape = this->inner_shape(shape);

      detail::new_executor_traits_detail::bulk_continuation_executor_adaptor<outer_executor_type> adapted_outer_executor(this->outer_executor());

      auto execute_me = bulk_then_execute_functor<Function,InnerFactory>{this->inner_executor(0), f, inner_shape, inner_factory};

      return adapted_outer_executor.bulk_then_execute(execute_me, outer_shape, predecessor, result_factory, outer_factory);
    }
};


} // end agency

