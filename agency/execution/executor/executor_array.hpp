#pragma once

#include <agency/detail/config.hpp>
#include <agency/experimental/ndarray.hpp>
#include <agency/detail/shape_tuple.hpp>
#include <agency/detail/index_tuple.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/memory/detail/unique_ptr.hpp>
#include <agency/execution/executor/detail/this_thread_parallel_executor.hpp>
#include <agency/execution/executor/detail/utility.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/executor/executor_traits/detail/member_barrier_type_or.hpp>
#include <agency/execution/executor/customization_points.hpp>
#include <agency/execution/executor/detail/execution_functions/bulk_then_execute.hpp>
#include <agency/execution/executor/properties/bulk_guarantee.hpp>
#include <agency/execution/executor/query.hpp>
#include <agency/detail/scoped_in_place_type.hpp>
#include <agency/tuple.hpp>


namespace agency
{


template<class InnerExecutor, class OuterExecutor = this_thread::parallel_executor>
class executor_array
{
  public:
    using outer_executor_type = OuterExecutor;
    using inner_executor_type = InnerExecutor;

  private:
    using outer_bulk_guarantee = decltype(bulk_guarantee_t::template static_query<outer_executor_type>());
    using inner_bulk_guarantee = decltype(bulk_guarantee_t::template static_query<inner_executor_type>());

    constexpr static size_t outer_depth = executor_execution_depth<outer_executor_type>::value;
    constexpr static size_t inner_depth = executor_execution_depth<inner_executor_type>::value;

  public:
    using outer_shape_type = executor_shape_t<outer_executor_type>;
    using inner_shape_type = executor_shape_t<inner_executor_type>;

    using outer_index_type = executor_index_t<outer_executor_type>;
    using inner_index_type = executor_index_t<inner_executor_type>;

    using shape_type = detail::scoped_shape_t<outer_depth,inner_depth,outer_shape_type,inner_shape_type>;
    using index_type = detail::scoped_index_t<outer_depth,inner_depth,outer_index_type,inner_index_type>;

    using barrier_type = detail::scoped_in_place_type_t_cat_t<
      detail::make_scoped_in_place_type_t<detail::member_barrier_type_or_t<outer_executor_type,void>>,
      detail::make_scoped_in_place_type_t<detail::member_barrier_type_or_t<inner_executor_type,void>>
    >;

    __AGENCY_ANNOTATION
    static shape_type make_shape(const outer_shape_type& outer_shape, const inner_shape_type& inner_shape)
    {
      return detail::make_scoped_shape<outer_depth,inner_depth>(outer_shape, inner_shape);
    }

    __agency_exec_check_disable__
    executor_array() = default;

    __agency_exec_check_disable__
    executor_array(const executor_array&) = default;

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    executor_array(size_t n, const inner_executor_type& exec = inner_executor_type())
      : inner_executors_(n, exec)
    {}

    template<class Iterator>
    executor_array(Iterator executors_begin, Iterator executors_end)
      : inner_executors_(executors_begin, executors_end)
    {}

    template<class T>
    using future = executor_future_t<outer_executor_type,T>;

    template<class T>
    using allocator = executor_allocator_t<outer_executor_type,T>;

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    friend constexpr bool operator==(const executor_array& a, const executor_array& b) noexcept
    {
      return a.outer_executor() == b.outer_executor() && a.inner_executors_ == b.inner_executors_;
    }

    __AGENCY_ANNOTATION
    friend constexpr bool operator!=(const executor_array& a, const executor_array& b) noexcept
    {
      return !(a == b);
    }

    // XXX this functor is public to allow nvcc to instantiate kernels with it
    template<class Futures, class UniquePtr1, class UniquePtr2>
    struct wait_for_futures_and_move_result
    {
      mutable Futures    futures_;
      mutable UniquePtr1 result_ptr_;
      mutable UniquePtr2 shared_arg_ptr_;

      __AGENCY_ANNOTATION
      typename std::pointer_traits<UniquePtr1>::element_type
        operator()() const
      {
        for(auto& f : futures_)
        {
          f.wait();
        }

        return std::move(*result_ptr_);
      }
    };

  private:
    template<class Futures, class UniquePtr1, class UniquePtr2>
    __AGENCY_ANNOTATION
    static wait_for_futures_and_move_result<typename std::decay<Futures>::type,UniquePtr1,UniquePtr2>
      make_wait_for_futures_and_move_result(Futures&& futures, UniquePtr1&& result_ptr, UniquePtr2&& shared_arg_ptr)
    {
      return wait_for_futures_and_move_result<typename std::decay<Futures>::type,UniquePtr1,UniquePtr2>{std::move(futures),std::move(result_ptr),std::move(shared_arg_ptr)};
    }

    __AGENCY_ANNOTATION
    static outer_shape_type outer_shape(const shape_type& shape)
    {
      // the outer portion is always the head of the tuple
      return __tu::tuple_head(shape);
    }

    __AGENCY_ANNOTATION
    static inner_shape_type inner_shape(const shape_type& shape)
    {
      // the inner portion of the shape is the tail of the tuple
      return detail::make_from_tail<inner_shape_type>(shape);
    }

    __AGENCY_ANNOTATION
    static index_type make_index(const outer_index_type& outer_idx, const inner_index_type& inner_idx)
    {
      return detail::make_scoped_index<outer_depth,inner_depth>(outer_idx, inner_idx);
    }

    __AGENCY_ANNOTATION
    size_t select_inner_executor(const outer_index_type& idx, const outer_shape_type& shape) const
    {
      size_t rank = detail::index_cast<size_t>(idx, shape, inner_executors_.size());
      
      // round robin through inner executors
      return rank % inner_executors_.size();
    }

    // lazy implementation of then_execute()
    // it is universally applicable, but it might not be as efficient
    // as calling execute() eagerly on the outer_executor
    struct lazy_strategy {};

    // eager implementation of then_execute()
    // not universally applicable, but can be more efficient
    // because work gets issued immediately to the outer_executor via execute()
    struct eager_strategy {};

    // XXX make this functor public to accomodate nvcc's requirement
    //     on types used to instantiate __global__ function templates
    template<class Function, class... InnerFactories>
    struct lazy_bulk_then_execute_functor
    {
      mutable executor_array exec;
      outer_shape_type outer_shape;
      inner_shape_type inner_shape;
      mutable Function f;
      tuple<InnerFactories...> inner_factories;

      template<class... OuterArgs>
      struct inner_functor
      {
        mutable Function f;
        outer_index_type outer_idx;
        tuple<OuterArgs&...> outer_args;

        template<size_t... Indices, class... InnerSharedArgs>
        __AGENCY_ANNOTATION
        void impl(detail::index_sequence<Indices...>, const inner_index_type& inner_idx, InnerSharedArgs&... inner_args) const
        {
          index_type idx = make_index(outer_idx, inner_idx);

          f(idx, agency::get<Indices>(outer_args)..., inner_args...);
        }

        template<class... InnerSharedArgs>
        __AGENCY_ANNOTATION
        void operator()(const inner_index_type& inner_idx, InnerSharedArgs&... inner_shared_args) const
        {
          impl(detail::make_index_sequence<sizeof...(OuterArgs)>(), inner_idx, inner_shared_args...);
        }
      };

      template<size_t... Indices, class... OuterArgs>
      __AGENCY_ANNOTATION
      void impl(detail::index_sequence<Indices...>, const outer_index_type& outer_idx, OuterArgs&... outer_args) const
      {
        auto inner_executor_idx = exec.select_inner_executor(outer_idx, outer_shape);
        inner_executor_type& inner_exec = exec.inner_executor(inner_executor_idx);

        // XXX avoid lambdas to workaround nvcc limitations
        //detail::blocking_bulk_twoway_execute_with_void_result(adapted_exec, [=,&predecessor,&result,&outer_shared_arg](const inner_index_type& inner_idx, detail::result_of_t<InnerFactories()>&... inner_shared_args)
        //{
        //  index_type idx = make_index(outer_idx, inner_idx);

        //  f(idx, predecessor, result, outer_shared_arg, inner_shared_args...);
        //},
        //inner_shape,
        //agency::get<Indices>(inner_factories)...);

        inner_functor<OuterArgs...> execute_me{f, outer_idx, agency::forward_as_tuple(outer_args...)};

        detail::blocking_bulk_twoway_execute_with_void_result(inner_exec, execute_me, inner_shape, agency::get<Indices>(inner_factories)...);
      }

      template<class... OuterArgs>
      __AGENCY_ANNOTATION
      void operator()(const outer_index_type& outer_idx, OuterArgs&... outer_args) const
      {
        impl(detail::make_index_sequence<sizeof...(InnerFactories)>(), outer_idx, outer_args...);
      }
    };

  private:
    template<class Function, class Future, class ResultFactory, class OuterFactory, class... InnerFactories>
    __AGENCY_ANNOTATION
    future<detail::result_of_t<ResultFactory()>>
      lazy_bulk_then_execute(Function f, shape_type shape, Future& predecessor, ResultFactory result_factory, OuterFactory outer_factory, InnerFactories... inner_factories) const
    {
      // this implementation of bulk_then_execute() is "lazy" in the sense that it
      // immediately calls bulk_then_execute() on the outer executor, but bulk_twoway_execute() is
      // called on the inner executors eventually at some point in the future

      // split shape into its outer and inner components
      outer_shape_type outer_shape = this->outer_shape(shape);
      inner_shape_type inner_shape = this->inner_shape(shape);

      // this commented-out code expressed with two lambdas is functionally equivalent to what happens with the named
      // functors below
      // XXX avoid lambdas to workaround nvcc limitations as well as lack of polymorphic lambda in c++11
      //return bulk_then_execute(outer_executor(), [=](const outer_index_type& outer_idx, auto&... outer_args)
      //{
      //  auto inner_executor_idx = select_inner_executor(outer_idx, outer_shape);

      //  blocking_bulk_twoway_execute_with_void_result(inner_executor(inner_executor_idx), [=](const inner_index_type& inner_idx, auto&... inner_args)
      //  {
      //    index_type idx = make_index(outer_idx, inner_idx);
      //    f(idx, outer_args..., inner_args...); 
      //  });
      //},
      //outer_shape,
      //predecessor,
      //result_factory,
      //outer_factory
      //);

      lazy_bulk_then_execute_functor<Function,InnerFactories...> execute_me{*this,outer_shape,inner_shape,f,agency::make_tuple(inner_factories...)};

      return detail::bulk_then_execute(outer_executor(), execute_me, outer_shape, predecessor, result_factory, outer_factory);
    }

    template<class Function, class Future, class ResultFactory, class OuterFactory, class... InnerFactories>
    __AGENCY_ANNOTATION
    future<detail::result_of_t<ResultFactory()>>
      bulk_then_execute_impl(lazy_strategy, Function f, shape_type shape, Future& predecessor, ResultFactory result_factory, OuterFactory outer_factory, InnerFactories... inner_factories) const
    {
      return lazy_bulk_then_execute(f, shape, predecessor, result_factory, outer_factory, inner_factories...);
    }

  public:
    // XXX make this functor public to accomodate nvcc's requirement
    //     on types used to instantiate __global__ function templates
    template<class Function, class Futures, class Result, class OuterShared, class... Factories>
    struct eager_bulk_then_execute_functor
    {
      const executor_array& exec;
      mutable Function f;
      Futures& predecessor_futures;
      Result* result_ptr;
      OuterShared* outer_shared_arg_ptr;
      tuple<Factories...> inner_factories;
      outer_shape_type outer_shape;
      inner_shape_type inner_shape;

      struct inner_functor
      {
        mutable Function f;
        outer_index_type outer_idx;
        Result& result;
        OuterShared& outer_arg;

        // this overload is chosen when the future is void
        // no additional past_arg will be passed to this function, so the number of Args
        // is equal to the number of shared Factories
        // unfortunately, this seems to be the most straightforward application of SFINAE
        // to choose between these two cases
        template<class... Args,
                 class = typename std::enable_if<
                   sizeof...(Args) == sizeof...(Factories)
                 >::type>
        __AGENCY_ANNOTATION
        void operator()(const inner_index_type& inner_idx, Args&... inner_shared_args) const
        {
          auto idx = make_index(outer_idx, inner_idx);

          // when the predecessor future is void, there's no predecessor argument to pass to invoke()
          agency::detail::invoke(f, idx, result, outer_arg, inner_shared_args...);
        }

        // this overload is chosen when the future is not void
        // an additional past_arg will be passed to this function, so the number of Args
        // unfortunately, this seems to be the most straightforward application of SFINAE
        // to choose between these two cases
        template<class Predecessor,
                 class... Args,
                 class = typename std::enable_if<
                   sizeof...(Args) == sizeof...(Factories)
                 >::type>
        __AGENCY_ANNOTATION
        void operator()(const inner_index_type& inner_idx, Predecessor& predecessor, Args&... inner_shared_args) const
        {
          auto idx = make_index(outer_idx, inner_idx);

          // when predecessor is not void, we pass it to invoke in the slot before result
          agency::detail::invoke(f, idx, predecessor, result, outer_arg, inner_shared_args...);
        }
      };

      template<size_t... Indices>
      __AGENCY_ANNOTATION
      executor_future_t<inner_executor_type,void>
        impl(detail::index_sequence<Indices...>, const outer_index_type& outer_idx) const
      {
        auto inner_executor_idx = exec.select_inner_executor(outer_idx, outer_shape);

        return detail::bulk_then_execute_with_void_result(
          exec.inner_executor(inner_executor_idx),
          inner_functor{f,outer_idx,*result_ptr,*outer_shared_arg_ptr},
          inner_shape,
          predecessor_futures[outer_idx],
          agency::get<Indices>(inner_factories)...
        );
      }

      __AGENCY_ANNOTATION
      executor_future_t<inner_executor_type,void>
        operator()(const outer_index_type& outer_idx) const
      {
        return impl(detail::index_sequence_for<Factories...>(), outer_idx);
      }
    };

  private:
    template<class Function, class Future, class ResultFactory, class OuterFactory, class... InnerFactories>
    __AGENCY_ANNOTATION
    future<detail::result_of_t<ResultFactory()>>
      eager_bulk_then_execute(Function f, shape_type shape, Future& predecessor, ResultFactory result_factory, OuterFactory outer_factory, InnerFactories... inner_factories) const
    {
      // this implementation legal when the outer_bulk_guarantee is not sequenced
      // XXX and the inner executor's is concurrent with the launching agent
      //     i.e., we have to make sure that the inner call to bulk_then_execute() actually makes progress
      //     without having to call .get() on the returned futures

      // separate the shape into inner and outer portions
      auto outer_shape = this->outer_shape(shape);
      auto inner_shape = this->inner_shape(shape);

      // create the results via the result_factory
      using result_type = decltype(result_factory());
      auto results_ptr = detail::allocate_unique<result_type>(allocator<result_type>(), result_factory());
      result_type* results_raw_ptr = results_ptr.get();

      // create the outer shared argument via the outer_factory
      using outer_shared_arg_type = decltype(outer_factory());
      auto outer_shared_arg_ptr = detail::allocate_unique<outer_shared_arg_type>(allocator<outer_shared_arg_type>(), outer_factory());
      outer_shared_arg_type* outer_shared_arg_raw_ptr = outer_shared_arg_ptr.get();

      // split the predecessor future into a collection of shared futures
      auto shared_predecessor_futures = detail::bulk_share_future(outer_executor(), outer_shape, predecessor);
      using future_container = decltype(shared_predecessor_futures);

      // XXX avoid lambdas to workaround nvcc limitations as well as lack of polymorphic lambda
      //auto inner_futures = blocking_bulk_twoway_execute_with_auto_result_and_without_shared_parameters(outer_executor(), [=,&past_futures](const outer_index_type& outer_idx) mutable
      //{
      //  auto inner_executor_idx = select_inner_executor(outer_idx, outer_shape);
      //
      //  using past_arg_type = future_result_t<Future>;
      //
      //  return bulk_then_execute_with_void_result(inner_executor(inner_executor_idx), [=](const inner_index_type& inner_idx, past_arg_type& past_arg, decltype(inner_factories())&... inner_shared_args) mutable
      //  {
      //    auto idx = make_index(outer_idx, inner_idx);
      //    (*results_raw_ptr)[idx] = agency::detail::invoke(f, idx, past_arg, *outer_shared_arg_raw_ptr, inner_shared_args...);
      //  },
      //  inner_shape,
      //  past_futures[outer_idx],
      //  inner_factories...);
      //},
      //outer_shape);

      auto functor = eager_bulk_then_execute_functor<Function,future_container,result_type,outer_shared_arg_type,InnerFactories...>{*this, f, shared_predecessor_futures, results_raw_ptr, outer_shared_arg_raw_ptr, agency::make_tuple(inner_factories...), outer_shape, inner_shape};
      auto inner_futures = detail::blocking_bulk_twoway_execute_with_auto_result_and_without_shared_parameters(outer_executor(), functor, outer_shape);

      // create a continuation to synchronize the futures and return the result
      auto continuation = make_wait_for_futures_and_move_result(std::move(inner_futures), std::move(results_ptr), std::move(outer_shared_arg_ptr));

      // twoway_execute() with the outer executor to launch the continuation
      return detail::twoway_execute(outer_executor(), std::move(continuation));
    }

    template<class Function, class Future, class ResultFactory, class OuterFactory, class... InnerFactories>
    __AGENCY_ANNOTATION
    future<detail::result_of_t<ResultFactory()>>
      bulk_then_execute_impl(eager_strategy, Function f, shape_type shape, Future& predecessor, ResultFactory result_factory, OuterFactory outer_factory, InnerFactories... inner_factories) const
    {
      return eager_bulk_then_execute(f, shape, predecessor, result_factory, outer_factory, inner_factories...);
    }

  public:
    template<class Function, class Future, class ResultFactory, class OuterFactory, class... InnerFactories,
             __AGENCY_REQUIRES(sizeof...(InnerFactories) == inner_depth)
            >
    __AGENCY_ANNOTATION
    future<detail::result_of_t<ResultFactory()>>
      bulk_then_execute(Function f, shape_type shape, Future& predecessor, ResultFactory result_factory, OuterFactory outer_factory, InnerFactories... inner_factories) const
    {
      // tag dispatch the appropriate implementation strategy for bulk_then_execute() using this first parameter
      return bulk_then_execute_impl(bulk_then_execute_implementation_strategy(), f, shape, predecessor, result_factory, outer_factory, inner_factories...);
    }

  private:
    outer_executor_type            outer_executor_;

    experimental::ndarray<inner_executor_type, 1, allocator<inner_executor_type>> inner_executors_;

    using bulk_then_execute_implementation_strategy = typename std::conditional<
      detail::disjunction<
        std::is_same<outer_bulk_guarantee, bulk_guarantee_t::sequenced_t>,
        std::is_same<inner_bulk_guarantee, bulk_guarantee_t::sequenced_t> // XXX this should really check whether the inner executor's twoway_execute() method executes concurrently with the caller 
      >::value,
      lazy_strategy,
      eager_strategy
    >::type;

    // XXX eliminate this when we eliminate .then_execute()
    using then_execute_implementation_strategy = bulk_then_execute_implementation_strategy;

  public:
    __AGENCY_ANNOTATION
    auto begin() ->
      decltype(inner_executors_.begin())
    {
      return inner_executors_.begin();
    }

    __AGENCY_ANNOTATION
    auto begin() const ->
      decltype(inner_executors_.cbegin())
    {
      return inner_executors_.cbegin();
    }

    __AGENCY_ANNOTATION
    auto end() ->
      decltype(inner_executors_.end())
    {
      return inner_executors_.end();
    }

    __AGENCY_ANNOTATION
    auto end() const ->
      decltype(inner_executors_.cend())
    {
      return inner_executors_.cend();
    }

    __AGENCY_ANNOTATION
    size_t size() const
    {
      return inner_executors_.size();
    }

    __AGENCY_ANNOTATION
    shape_type unit_shape() const
    {
      auto outer_exec_shape = size() * agency::unit_shape(outer_executor());
      auto inner_exec_shape = agency::unit_shape(inner_executor(0));

      return make_shape(outer_exec_shape, inner_exec_shape);
    }

    __AGENCY_ANNOTATION
    shape_type max_shape_dimensions() const
    {
      // XXX might want to multiply shape() * max_shape_dimensions(outer_executor())
      auto outer_max_shape = agency::max_shape_dimensions(outer_executor());

      auto inner_max_shape = agency::max_shape_dimensions(inner_executor(0));

      return make_shape(outer_max_shape, inner_max_shape);
    }

    __AGENCY_ANNOTATION
    outer_executor_type& outer_executor()
    {
      return outer_executor_;
    }

    __AGENCY_ANNOTATION
    const outer_executor_type& outer_executor() const
    {
      return outer_executor_;
    }

    __AGENCY_ANNOTATION
    inner_executor_type& inner_executor(size_t i)
    {
      return begin()[i];
    }

    __AGENCY_ANNOTATION
    const inner_executor_type& inner_executor(size_t i) const
    {
      return begin()[i];
    }

    __AGENCY_ANNOTATION
    inner_executor_type& operator[](size_t i)
    {
      return inner_executors_[i];
    }

    __AGENCY_ANNOTATION
    const inner_executor_type& operator[](size_t i) const
    {
      return inner_executors_[i];
    }

    __AGENCY_ANNOTATION
    constexpr static bulk_guarantee_t::scoped_t<outer_bulk_guarantee, inner_bulk_guarantee>
      query(const bulk_guarantee_t&)
    {
      return bulk_guarantee_t::scoped(outer_bulk_guarantee{}, inner_bulk_guarantee{});
    }
};


} // end agency

