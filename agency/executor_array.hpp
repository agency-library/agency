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
    template<class Futures, class Result, class Alloc>
    struct wait_for_futures_and_move_result
    {
      mutable Futures futures_;
      mutable agency::detail::unique_ptr<Result,Alloc> result_ptr_;

      __AGENCY_ANNOTATION
      Result operator()() const
      {
        for(auto& f : futures_)
        {
          f.wait();
        }

        return std::move(*result_ptr_);
      }
    };

    template<class Futures, class Result, class Deleter>
    __AGENCY_ANNOTATION
    wait_for_futures_and_move_result<typename std::decay<Futures>::type,Result,Deleter>
      make_wait_for_futures_and_move_result(Futures&& futures, agency::detail::unique_ptr<Result,Deleter>&& result_ptr)
    {
      return wait_for_futures_and_move_result<typename std::decay<Futures>::type,Result,Deleter>{std::move(futures),std::move(result_ptr)};
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
    template<class Function, class Factory1>
    future<typename std::result_of<Factory1(shape_type)>::type>
      async_execute(Function f, Factory1 result_factory, shape_type shape)
    {
      auto outer_shape = this->outer_shape(shape);
      auto inner_shape = this->inner_shape(shape);

      using result_type = decltype(result_factory(shape));
      auto results_ptr = detail::allocate_unique<result_type>(allocator<result_type>(), result_factory(shape));
      result_type *results_raw_ptr = results_ptr.get();

      auto futures = outer_traits::execute(outer_executor(), [=](const outer_index_type& outer_idx)
      {
        auto inner_executor_idx = select_inner_executor(outer_idx, outer_shape);

        // XXX won't outer_shared_arg go out of scope by the time this executes?
        //     we ought to create the outer_shared_arg with the results
        return inner_traits::async_execute(inner_executor(inner_executor_idx), [=](const inner_index_type& inner_idx)
        {
          auto idx = make_index(outer_idx, inner_idx);
          (*results_raw_ptr)[idx] = agency::invoke(f, idx);
        },
        inner_shape);
      },
      outer_shape);

      auto continuation = make_wait_for_futures_and_move_result(std::move(futures), std::move(results_ptr));
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

