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

    template<class T>
    using future = typename outer_traits::template future<T>;

    template<class T>
    using allocator = typename outer_traits::template allocator<T>;

    template<class T>
    using container = agency::detail::array<T, shape_type, allocator<T>>;

  private:
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
      // the inner portion is the tail of the tuple, but if the 
      // inner executor is not nested, then the tuple needs to be unwrapped
      return detail::unwrap_tuple_if_not_nested<inner_execution_category>(detail::forward_tail(shape));
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

  public:
    //template<class Function, class Factory1, class Factory2, class... Factories,
    //         class = typename std::enable_if<
    //           sizeof...(Factories) == inner_depth
    //         >::type>
    //future<typename std::result_of<Factory1(shape_type)>::type>
    //  async_execute(Function f, Factory1 result_factory, shape_type shape, Factory2 outer_factory, Factories... inner_factories)
    //{
    //  auto outer_shape = outer_shape(shape);
    //  auto inner_shape = inner_shape(shape);

    //  using result_type = decltype(result_factory(shape));
    //  auto results_ptr = detail::allocate_unique<result_type>(allocator<result_type>(), result_factory(shape));
    //  result_type *results_raw_ptr = results_ptr.get();

    //  auto futures = outer_traits::execute(outer_executor(), [=](const outer_index_type& outer_idx, auto& outer_shared_arg)
    //  {
    //    auto inner_executor_idx = select_inner_executor(outer_idx, outer_shape);

    //    // XXX won't outer_shared_arg go out of scope by the time this executes?
    //    //     we ought to create the outer_shared_arg with the results
    //    return inner_traits::async_execute(inner_executor(inner_executor_idx), [=outer_idx,&results,&outer_shared_arg](const inner_index_type& inner_idx, auto&... inner_shared_args)
    //    {
    //      auto idx = make_index(outer_idx, inner_idx);
    //      (*results_raw_ptr)[idx] = agency::invoke(f, idx, outer_shared_arg, inner_shared_args...);
    //    },
    //    inner_shape,
    //    inner_factories...);
    //  },
    //  outer_shape,
    //  outer_factory);

    //  auto continuation = make_wait_for_futures_and_move_result(std::move(futures), std::move(results_ptr));
    //  return outer_traits::then_execute(outer_executor(), std::move(continuation));
    //}

    // XXX this implementation is only valid for outer_execution_category != sequential_execution_tag
    // XXX generalize this to support shared parameters
    template<class Function, class Factory1, class Factory2>
    future<typename std::result_of<Factory1(shape_type)>::type>
      async_execute(Function f, Factory1 result_factory, shape_type shape, Factory2 outer_factory)
    {
      auto outer_shape = this->outer_shape(shape);
      auto inner_shape = this->inner_shape(shape);

      // create the results via the result_factory
      using result_type = decltype(result_factory(shape));
      auto results_ptr = detail::allocate_unique(allocator<result_type>(), result_factory(shape));
      result_type* results_raw_ptr = results_ptr.get();

      // create the outer shared argument via the outer_factory
      using outer_shared_arg_type = decltype(outer_factory());
      auto outer_shared_arg_ptr = detail::allocate_unique(allocator<outer_shared_arg_type>(), outer_factory());
      outer_shared_arg_type* outer_shared_arg_raw_ptr = outer_shared_arg_ptr.get();

      // eagerly execute() with the outer executor so we can issue these async_execute() calls immediately
      auto futures = outer_traits::execute(outer_executor(), [=](const outer_index_type& outer_idx)
      {
        auto inner_executor_idx = select_inner_executor(outer_idx, outer_shape);

        // XXX won't outer_shared_arg go out of scope by the time this executes?
        //     we ought to create the outer_shared_arg with the results
        return inner_traits::async_execute(inner_executor(inner_executor_idx), [=](const inner_index_type& inner_idx)
        {
          auto idx = make_index(outer_idx, inner_idx);
          (*results_raw_ptr)[idx] = agency::invoke(f, idx, *outer_shared_arg_raw_ptr);
        },
        inner_shape);
      },
      outer_shape);

      // create a continuation to synchronize the futures and return the result
      auto continuation = make_wait_for_futures_and_move_result(std::move(futures), std::move(results_ptr), std::move(outer_shared_arg_ptr));

      // async_execute() with the outer executor to launch the continuation
      return outer_traits::async_execute(outer_executor(), std::move(continuation));
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
      decltype(inner_executors_.begin())
    {
      return inner_executors_.begin();
    }

    __AGENCY_ANNOTATION
    auto end() ->
      decltype(inner_executors_.end())
    {
      return inner_executors_.end();
    }

    __AGENCY_ANNOTATION
    auto end() const ->
      decltype(inner_executors_.end())
    {
      return inner_executors_.end();
    }

    __AGENCY_ANNOTATION
    size_t size() const
    {
      return inner_executors_.size();
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
    inner_executor_type& inner_executor(size_t i) const
    {
      return begin()[i];
    }
};


} // end agency

