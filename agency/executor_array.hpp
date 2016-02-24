#pragma once

#include <agency/parallel_executor.hpp>
#include <agency/detail/array.hpp>
#include <agency/detail/shape_tuple.hpp>
#include <agency/detail/index_tuple.hpp>
#include <agency/functional.hpp>

namespace agency
{


template<class InnerExecutor, class OuterExecutor = this_thread::parallel_executor>
class executor_array
{
  public:
    using outer_executor_type = OuterExecutor;
    using inner_executor_type = InnerExecutor;

  private:
    using inner_traits = executor_traits<inner_executor_type>;
    using outer_traits = executor_traits<outer_executor_type>;

    using outer_execution_category = typename outer_traits::execution_category;
    using inner_execution_category = typename inner_traits::execution_category;

    constexpr static size_t inner_depth = inner_traits::execution_depth;

  public:
    using execution_category = nested_execution_tag<outer_execution_category,inner_execution_category>;

    using outer_shape_type = typename outer_traits::shape_type;
    using inner_shape_type = typename inner_traits::shape_type;

    using outer_index_type = typename outer_traits::index_type;
    using inner_index_type = typename inner_traits::index_type;

    using shape_type = detail::nested_shape_t<outer_execution_category,inner_execution_category,outer_shape_type,inner_shape_type>;
    using index_type = detail::nested_index_t<outer_execution_category,inner_execution_category,outer_index_type,inner_index_type>;

    __AGENCY_ANNOTATION
    static shape_type make_shape(const outer_shape_type& outer_shape, const inner_shape_type& inner_shape)
    {
      return detail::make_nested_shape<outer_execution_category,inner_execution_category>(outer_shape, inner_shape);
    }

    __AGENCY_ANNOTATION
    executor_array() = default;

    __AGENCY_ANNOTATION
    executor_array(const executor_array&) = default;

    __AGENCY_ANNOTATION
    executor_array(size_t n, const inner_executor_type& exec = inner_executor_type())
      : inner_executors_(n, exec)
    {}

    template<class Iterator>
    executor_array(Iterator executors_begin, Iterator executors_end)
      : inner_executors_(executors_begin, executors_end)
    {}

    template<class T>
    using future = typename outer_traits::template future<T>;

    template<class T>
    using allocator = typename outer_traits::template allocator<T>;

    template<class T>
    using container = agency::detail::array<T, shape_type, allocator<T>, index_type>;

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
    wait_for_futures_and_move_result<typename std::decay<Futures>::type,UniquePtr1,UniquePtr2>
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
      return detail::make_nested_index<outer_execution_category,inner_execution_category>(outer_idx, inner_idx);
    }

    __AGENCY_ANNOTATION
    size_t select_inner_executor(const outer_index_type& idx, const outer_shape_type& shape) const
    {
      size_t rank = detail::index_cast<size_t>(idx, shape, inner_executors_.size());
      
      // round robin through inner executors
      return rank % inner_executors_.size();
    }

  // XXX make this functor public to accomodate nvcc's requirement
  //     on types used to instantiate __global__ function templates
  public:
    template<class Function, class... Factories>
    struct then_execute_sequential_functor
    {
      executor_array& exec;
      mutable Function f;
      detail::tuple<Factories...> inner_factories;
      outer_shape_type outer_shape;
      inner_shape_type inner_shape;

      template<class ResultsType, class PastArgType, class OuterSharedArgType>
      struct inner_functor_with_past_arg
      {
        mutable Function f;
        outer_index_type outer_idx;
        ResultsType& results;
        PastArgType& past_arg;
        OuterSharedArgType& outer_shared_arg;

        template<class... Args>
        __AGENCY_ANNOTATION
        void operator()(const inner_index_type& inner_idx, Args&... inner_shared_args)
        {
          auto idx = make_index(outer_idx, inner_idx);
          results[idx] = agency::invoke(f, idx, past_arg, outer_shared_arg, inner_shared_args...);
        }
      };

      template<class ResultsType, class OuterSharedArgType>
      struct inner_functor_without_past_arg
      {
        mutable Function f;
        outer_index_type outer_idx;
        ResultsType& results;
        OuterSharedArgType& outer_shared_arg;

        template<class... Args>
        void operator()(const inner_index_type& inner_idx, Args&... inner_shared_args)
        {
          auto idx = make_index(outer_idx, inner_idx);
          results[idx] = agency::invoke(f, idx, outer_shared_arg, inner_shared_args...);
        }
      };

      template<size_t... Indices, class... Args>
      __AGENCY_ANNOTATION
      void unpack_factories_and_call_execute(detail::index_sequence<Indices...>, inner_executor_type& exec, Args&&... args)
      {
        inner_traits::execute(exec, std::forward<Args>(args)..., detail::get<Indices>(inner_factories)...);
      }

      template<class ResultsType, class PastArgType, class OuterSharedArgType>
      __AGENCY_ANNOTATION
      void operator()(const outer_index_type& outer_idx, ResultsType& results, PastArgType& past_arg, OuterSharedArgType& outer_shared_arg)
      {
        auto inner_executor_idx = exec.select_inner_executor(outer_idx, outer_shape);
        auto& inner_exec = exec.inner_executor(inner_executor_idx);

        auto functor = inner_functor_with_past_arg<ResultsType,PastArgType,OuterSharedArgType>{f, outer_idx, results, past_arg, outer_shared_arg};

        // execute the functor on the inner executor
        unpack_factories_and_call_execute(detail::index_sequence_for<Factories...>(), inner_exec, functor, inner_shape);
      }

      template<class ResultsType, class OuterSharedArgType>
      __AGENCY_ANNOTATION
      void operator()(const outer_index_type& outer_idx, ResultsType& results, OuterSharedArgType& outer_shared_arg)
      {
        auto inner_executor_idx = exec.select_inner_executor(outer_idx, outer_shape);
        auto& inner_exec = exec.inner_executor(inner_executor_idx);

        auto functor = inner_functor_without_past_arg<ResultsType,OuterSharedArgType>{f, outer_idx, results, outer_shared_arg};

        // execute the functor on the inner executor
        unpack_factories_and_call_execute(detail::index_sequence_for<Factories...>(), inner_exec, functor, inner_shape);
      }
    };

  private:
    // implementation of then_execute() for sequential OuterExecutors
    template<class Function, class Factory1, class Future, class Factory2, class... Factories,
             class = typename std::enable_if<
               sizeof...(Factories) == inner_depth
             >::type>
    future<typename std::result_of<Factory1(shape_type)>::type>
      then_execute_impl(std::true_type, Function f, Factory1 result_factory, shape_type shape, Future& fut, Factory2 outer_factory, Factories... inner_factories)
    {
      // separate the shape into inner and outer portions
      auto outer_shape = this->outer_shape(shape);
      auto inner_shape = this->inner_shape(shape);

      // create the results via the result_factory
      using result_type = decltype(result_factory(shape));
      auto results_fut = outer_traits::template make_ready_future<result_type>(outer_executor(), result_factory(shape));
      auto futures = agency::detail::make_tuple(std::move(results_fut), std::move(fut));

      // XXX doesn't work when past_arg_type is void
      using past_arg_type = typename future_traits<Future>::value_type;
      using outer_shared_arg_type = decltype(outer_factory());

      // XXX avoid lambdas to workaround nvcc limitations as well as lack of polymorphic lambda
      //return outer_traits::template when_all_execute_and_select<0>(outer_executor(), [=](const outer_index_type& outer_idx, result_type& results, past_arg_type& past_arg, outer_shared_arg_type& outer_shared_arg) mutable
      //{
      //  auto inner_executor_idx = select_inner_executor(outer_idx, outer_shape);

      //  inner_traits::execute(inner_executor(inner_executor_idx), [=,&outer_shared_arg,&results](const inner_index_type& inner_idx, decltype(inner_factories())&... inner_shared_args) mutable
      //  {
      //    auto idx = make_index(outer_idx, inner_idx);
      //    results[idx] = agency::invoke(f, idx, past_arg, outer_shared_arg, inner_shared_args...);
      //  },
      //  inner_shape,
      //  inner_factories...);
      //},
      //outer_shape,
      //futures,
      //outer_factory);

      auto functor = then_execute_sequential_functor<Function,Factories...>{*this, f, detail::make_tuple(inner_factories...), outer_shape, inner_shape};
      return outer_traits::template when_all_execute_and_select<0>(outer_executor(), functor, outer_shape, futures, outer_factory);
      
      // XXX another option:
      // call a chain of then_execute()s, each then_execute() stores its results into the appropriate slots of the result
      // the last then_execute() would return all the results, somehow
    }

  public:
    // XXX make this functor public to accomodate nvcc's requirement
    //     on types used to instantiate __global__ function templates
    template<class Function, class Results, class Futures, class OuterShared, class... Factories>
    struct then_execute_non_sequential_functor
    {
      executor_array& exec;
      mutable Function f;
      Results* results_ptr;
      Futures& past_futures;
      OuterShared* outer_shared_arg_ptr;
      detail::tuple<Factories...> inner_factories;
      outer_shape_type outer_shape;
      inner_shape_type inner_shape;

      struct inner_functor
      {
        mutable Function f;
        outer_index_type outer_idx;
        Results& results;
        OuterShared& outer_arg;

        template<class... Args>
        __AGENCY_ANNOTATION
        void operator()(const inner_index_type& inner_idx, Args&... inner_shared_args) const
        {
          auto idx = make_index(outer_idx, inner_idx);

          results[idx] = agency::invoke(f, idx, outer_arg, inner_shared_args...);
        }
      };

      template<size_t... Indices>
      __AGENCY_ANNOTATION
      typename inner_traits::template future<void>
        impl(detail::index_sequence<Indices...>, const outer_index_type& outer_idx) const
      {
        auto inner_executor_idx = exec.select_inner_executor(outer_idx, outer_shape);

        return inner_traits::then_execute(
          exec.inner_executor(inner_executor_idx),
          inner_functor{f,outer_idx,*results_ptr,*outer_shared_arg_ptr},
          inner_shape,
          past_futures[outer_idx],
          detail::get<Indices>(inner_factories)...
        );
      }

      __AGENCY_ANNOTATION
      typename inner_traits::template future<void>
        operator()(const outer_index_type& outer_idx) const
      {
        return impl(detail::index_sequence_for<Factories...>(), outer_idx);
      }
    };

  private:
    // implementation of then_execute() for nonsequential OuterExecutors
    template<class Function, class Factory1, class Future, class Factory2, class... Factories,
             class = typename std::enable_if<
               sizeof...(Factories) == inner_depth
             >::type>
    future<typename std::result_of<Factory1(shape_type)>::type>
      then_execute_impl(std::false_type, Function f, Factory1 result_factory, shape_type shape, Future& fut, Factory2 outer_factory, Factories... inner_factories)
    {
      // this implementation legal when the outer_category is not sequential

      // separate the shape into inner and outer portions
      auto outer_shape = this->outer_shape(shape);
      auto inner_shape = this->inner_shape(shape);

      // create the results via the result_factory
      using result_type = decltype(result_factory(shape));
      auto results_ptr = detail::allocate_unique<result_type>(allocator<result_type>(), result_factory(shape));
      result_type* results_raw_ptr = results_ptr.get();

      // create the outer shared argument via the outer_factory
      using outer_shared_arg_type = decltype(outer_factory());
      auto outer_shared_arg_ptr = detail::allocate_unique<outer_shared_arg_type>(allocator<outer_shared_arg_type>(), outer_factory());
      outer_shared_arg_type* outer_shared_arg_raw_ptr = outer_shared_arg_ptr.get();

      using past_arg_type = typename future_traits<Future>::value_type;

      // split the incoming future into a collection of shared futures
      auto past_futures = outer_traits::share_future(outer_executor(), fut, outer_shape);
      using future_container = decltype(past_futures);

      // XXX avoid lambdas to workaround nvcc limitations as well as lack of polymorphic lambda
      //auto inner_futures = outer_traits::execute(outer_executor(), [=,&past_futures](const outer_index_type& outer_idx) mutable
      //{
      //  auto inner_executor_idx = select_inner_executor(outer_idx, outer_shape);

      //  return inner_traits::then_execute(inner_executor(inner_executor_idx), [=](const inner_index_type& inner_idx, past_arg_type& past_arg, decltype(inner_factories())&... inner_shared_args) mutable
      //  {
      //    auto idx = make_index(outer_idx, inner_idx);
      //    (*results_raw_ptr)[idx] = agency::invoke(f, idx, past_arg, *outer_shared_arg_raw_ptr, inner_shared_args...);
      //  },
      //  inner_shape,
      //  past_futures[outer_idx],
      //  inner_factories...);
      //},
      //outer_shape);

      auto functor = then_execute_non_sequential_functor<Function,result_type,future_container,outer_shared_arg_type,Factories...>{*this, f, results_raw_ptr, past_futures, outer_shared_arg_raw_ptr, detail::make_tuple(inner_factories...), outer_shape, inner_shape};
      auto inner_futures = outer_traits::execute(outer_executor(), functor, outer_shape);

      // create a continuation to synchronize the futures and return the result
      auto continuation = make_wait_for_futures_and_move_result(std::move(inner_futures), std::move(results_ptr), std::move(outer_shared_arg_ptr));

      // async_execute() with the outer executor to launch the continuation
      return outer_traits::async_execute(outer_executor(), std::move(continuation));
    }

  public:
    template<class Function, class Factory1, class Future, class Factory2, class... Factories,
             class = typename std::enable_if<
               sizeof...(Factories) == inner_depth
             >::type>
    future<typename std::result_of<Factory1(shape_type)>::type>
      then_execute(Function f, Factory1 result_factory, shape_type shape, Future& fut, Factory2 outer_factory, Factories... inner_factories)
    {
      using outer_execution_is_sequential = std::is_same<
        outer_execution_category,
        sequential_execution_tag
      >;

      return then_execute_impl(outer_execution_is_sequential(), f, result_factory, shape, fut, outer_factory, inner_factories...);
    }

  private:
    outer_executor_type            outer_executor_;

    // XXX consider using container here instead of array
    agency::detail::array<inner_executor_type, size_t, allocator<inner_executor_type>> inner_executors_;

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
    shape_type shape() const
    {
      auto outer_exec_shape = size() * outer_traits::shape(outer_executor());
      auto inner_exec_shape = inner_traits::shape(inner_executor(0));

      return make_shape(outer_exec_shape, inner_exec_shape);
    }

    __AGENCY_ANNOTATION
    shape_type max_shape_dimensions() const
    {
      // XXX might want to multiply shape() * max_shape_dimensions(outer_executor())
      auto outer_max_shape = outer_traits::max_shape_dimensions(outer_executor());

      auto inner_max_shape = inner_traits::max_shape_dimensions(inner_executor(0));

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
};


} // end agency

