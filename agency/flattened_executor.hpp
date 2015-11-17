#pragma once

#include <type_traits>
#include <agency/detail/tuple.hpp>
#include <agency/executor_traits.hpp>
#include <agency/execution_categories.hpp>
#include <agency/nested_executor.hpp>
#include <agency/detail/factory.hpp>
#include <agency/detail/optional.hpp>

namespace agency
{
namespace detail
{


// XXX we should find a better home for this functionality because grid_executor.hpp replicates this code
template<class Container>
struct guarded_container : Container
{
  using Container::Container;

  __AGENCY_ANNOTATION
  guarded_container(Container&& other)
    : Container(std::move(other))
  {}

  template<class Index>
  struct reference
  {
    Container& self_;
    Index idx_;

    template<class Optional>
    __AGENCY_ANNOTATION
    void operator=(Optional&& opt)
    {
      if(opt)
      {
        self_[idx_] = std::forward<Optional>(opt).value();
      }
    }
  };

  template<class Index>
  __AGENCY_ANNOTATION
  reference<Index> operator[](const Index& idx)
  {
    return reference<Index>{*this,idx};
  }
};


template<class Container>
__AGENCY_ANNOTATION
guarded_container<typename std::decay<Container>::type> make_guarded_container(Container&& value)
{
  return guarded_container<typename std::decay<Container>::type>(std::forward<Container>(value));
}


template<class Factory, class Shape>
struct guarded_container_factory
{
  Factory factory_;
  Shape shape_;

  using container_type = typename std::result_of<Factory(Shape)>::type;

  __agency_hd_warning_disable__
  template<class Arg>
  __AGENCY_ANNOTATION
  guarded_container<container_type> operator()(const Arg&)
  {
    return make_guarded_container(factory_(shape_));
  }
};


} // end detail


template<class Executor>
class flattened_executor
{
  // probably shouldn't insist on a nested executor
  static_assert(
    detail::is_nested_execution_category<typename executor_traits<Executor>::execution_category>::value,
    "Execution category of Executor must be nested."
  );

  public:
    // XXX what is the execution category of a flattened executor?
    using execution_category = parallel_execution_tag;

    using base_executor_type = Executor;

    // XXX maybe use whichever of the first two elements of base_executor_type::shape_type has larger dimensionality?
    using shape_type = size_t;

    template<class T>
    using future = typename executor_traits<base_executor_type>::template future<T>;

    future<void> make_ready_future()
    {
      return executor_traits<base_executor_type>::make_ready_future(base_executor());
    }

    flattened_executor(const base_executor_type& base_executor = base_executor_type())
      : min_inner_size_(1000),
        outer_subscription_(std::max(1u, log2(std::max(1u,std::thread::hardware_concurrency())))),
        base_executor_(base_executor)
    {}

    template<class Function, class Factory1, class Future, class Factory2>
    future<typename std::result_of<Factory1(shape_type)>::type>
      then_execute(Function f, Factory1 result_factory, shape_type shape, Future& dependency, Factory2 shared_factory)
    {
      auto partitioning = partition(shape);

      // store results into an intermediate result
      detail::guarded_container_factory<Factory1,shape_type> intermediate_result_factory{result_factory,shape};

      // execute
      auto intermediate_fut = executor_traits<base_executor_type>::then_execute(base_executor(), then_execute_functor<Function>{f, shape, partitioning}, intermediate_result_factory, partitioning, dependency, shared_factory, agency::detail::unit_factory());

      // cast the intermediate result to the type of result expected by the caller
      using result_type = typename std::result_of<Factory1(shape_type)>::type;
      return executor_traits<base_executor_type>::template future_cast<result_type>(base_executor(), intermediate_fut);
    }

    const base_executor_type& base_executor() const
    {
      return base_executor_;
    }

    base_executor_type& base_executor()
    {
      return base_executor_;
    }

  private:
    using partition_type = typename executor_traits<base_executor_type>::shape_type;

    template<class Function>
    struct then_execute_functor
    {
      Function f;
      shape_type shape;
      partition_type partitioning;

      // the result of this functor is void when the result of f(...) is void
      // when the result of f(...) is any other type T, the result is optional<T>
      template<class T>
      using result_t = typename std::conditional<
        std::is_void<T>::value,
        void,
        detail::optional<T>
      >::type;

      template<class Index>
      __AGENCY_ANNOTATION
      auto flatten_index(const Index& idx) const
        -> decltype(
             agency::detail::get<0>(idx) * agency::detail::get<1>(partitioning) + agency::detail::get<1>(idx)
           )
      {
        return agency::detail::get<0>(idx) * agency::detail::get<1>(partitioning) + agency::detail::get<1>(idx);
      }

      template<class Index>
      __AGENCY_ANNOTATION
      result_t<typename std::result_of<Function(Index)>::type>
        operator()(const Index& idx)
      {
        auto flat_idx = flatten_index(idx);

        if(flat_idx < shape)
        {
          return f(flat_idx);
        }

        using result_type = result_t<typename std::result_of<Function(Index)>::type>;
        return static_cast<result_type>(detail::nullopt);
      }

      template<class Index, class Arg>
      __AGENCY_ANNOTATION
      result_t<typename std::result_of<Function(Index,Arg&)>::type>
        operator()(const Index& idx, Arg& arg, agency::detail::unit)
      {
        auto flat_idx = flatten_index(idx);

        if(flat_idx < shape)
        {
          return f(flat_idx, arg);
        }

        using result_type = result_t<typename std::result_of<Function(Index,Arg&)>::type>;
        return static_cast<result_type>(detail::nullopt);
      }

      template<class Index, class Arg1, class Arg2>
      __AGENCY_ANNOTATION
      result_t<typename std::result_of<Function(Index,Arg1&,Arg2&)>::type>
        operator()(const Index& idx, Arg1& arg1, Arg2& arg2, agency::detail::unit)
      {
        auto flat_idx = flatten_index(idx);

        if(flat_idx < shape)
        {
          return f(flat_idx, arg1, arg2);
        }

        using result_type = result_t<typename std::result_of<Function(Index,Arg1&,Arg2&)>::type>;
        return static_cast<result_type>(detail::nullopt);
      }
    };

    // returns (outer size, inner size)
    partition_type partition(shape_type shape) const
    {
      // avoid division by zero below
      // XXX implement me!
//      if(is_empty(shape)) return partition_type{};

      using outer_shape_type = typename std::tuple_element<0,partition_type>::type;
      using inner_shape_type = typename std::tuple_element<1,partition_type>::type;

      outer_shape_type outer_size = (shape + min_inner_size_ - 1) / min_inner_size_;

      outer_size = std::min<size_t>(outer_subscription_ * std::thread::hardware_concurrency(), outer_size);

      inner_shape_type inner_size = (shape + outer_size - 1) / outer_size;

      return partition_type{outer_size, inner_size};
    }

    inline static unsigned int log2(unsigned int x)
    {
      unsigned int result = 0;
      while(x >>= 1) ++result;
      return result;
    }

    size_t min_inner_size_;
    size_t outer_subscription_;
    base_executor_type base_executor_;
};


} // end agency

