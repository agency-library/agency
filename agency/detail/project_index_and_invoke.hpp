#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/factory.hpp>
#include <agency/detail/optional.hpp>
#include <agency/detail/index_cast.hpp>
#include <agency/detail/shape_cast.hpp>
#include <utility>
#include <type_traits>

namespace agency
{
namespace detail
{


template<class Index, class Function, class Shape>
struct project_index_and_invoke
{
  using index_type = Index;
  using shape_type = Shape;

  using projected_index_type = decltype(
    detail::project_index(std::declval<index_type>(), std::declval<shape_type>())
  );

  using projected_shape_type  = decltype(
    project_shape(std::declval<shape_type>())
  );

  mutable Function     f_;
  shape_type           shape_;
  projected_shape_type projected_shape_;

  __AGENCY_ANNOTATION
  project_index_and_invoke(const Function& f, shape_type shape, projected_shape_type projected_shape)
    : f_(f),
      shape_(shape),
      projected_shape_(projected_shape)
  {}

  // this type stores the result of f_(index)
  template<class T>
  struct value_and_index
  {
    T value;
    projected_index_type index;
  };

  template<class T>
  __AGENCY_ANNOTATION
  value_and_index<typename std::decay<T>::type> make_value_and_index(T&& value, projected_index_type idx) const
  {
    return value_and_index<typename std::decay<T>::type>{std::forward<T>(value), idx};
  }

  // this is the type of result returned by f_
  template<class... Args>
  using result_of_function_t = typename std::result_of<Function(projected_index_type,Args...)>::type;

  template<class T>
  using void_or_optionally_value_and_index_t = typename std::conditional<
    std::is_void<T>::value,
    void,
    optional<value_and_index<T>>
  >::type;

  // this is the type of result returned by this functor
  template<class... Args>
  using result_t = void_or_optionally_value_and_index_t<result_of_function_t<Args...>>;

  template<size_t... Indices>
  __AGENCY_ANNOTATION
  projected_index_type project_index_impl(detail::index_sequence<Indices...>, const Index& idx) const
  {
    // to project a multidimensional index, we take the first two elements of idx and merge them together
    // this merger becomes the first element of the result
    // the remaining elements of idx shift one position left
    return projected_index_type(detail::get<1>(idx) + detail::get<0>(idx) * detail::get<1>(shape_), detail::get<2 + Indices>(idx)...);
  }

  // XXX WAR nvcc issue with handling empty index_sequence<> given to the function above
  __AGENCY_ANNOTATION
  projected_index_type project_index_impl(detail::index_sequence<>, const Index& idx) const
  {
    return projected_index_type(detail::get<1>(idx) + detail::get<0>(idx) * detail::get<1>(shape_));
  }

  __AGENCY_ANNOTATION
  projected_index_type project_index(const Index& idx) const
  {
    return project_index_impl(detail::make_index_sequence<std::tuple_size<Index>::value - 2>(), idx);
  }

  // when f_(idx) has no result, we just return void
  template<class... Args,
           class = typename std::enable_if<
             std::is_void<
               result_of_function_t<Args&&...>
             >::value
           >::type
          >
  __AGENCY_ANNOTATION
  void impl(const Index& idx, Args&&... args) const
  {
    auto projected_idx = this->project_index(idx);

    if(projected_idx < projected_shape_)
    {
      f_(projected_idx, std::forward<Args>(args)...);
    }
  }

  // when f_(idx) has a result, we optionally return f_'s result and the index,
  // but only at points where f_(idx) is defined
  template<class... Args,
           class = typename std::enable_if<
             !std::is_void<
               result_of_function_t<Args&&...>
             >::value
           >::type
          >
  __AGENCY_ANNOTATION
  result_t<Args&&...>
    impl(const Index& idx, Args&&... args) const
  {
    auto projected_idx = this->project_index(idx);

    if(projected_idx < projected_shape_)
    {
      return make_value_and_index(f_(projected_idx, std::forward<Args>(args)...), projected_idx);
    }

    return nullopt;
  }

  // this overload implements the functor for then_execute() when the dependency future is void
  template<class T>
  __AGENCY_ANNOTATION
  result_t<T&>
    operator()(const Index& idx, T& outer_shared_parameter, unit) const
  {
    return impl(idx, outer_shared_parameter);
  }

  // this overload implements the functor for then_execute() when the dependency future is not void
  template<class T1, class T2>
  __AGENCY_ANNOTATION
  result_t<T1&,T2&>
    operator()(const Index& idx, T1& past_parameter, T2& outer_shared_parameter, unit) const
  {
    return impl(idx, past_parameter, outer_shared_parameter);
  }
};


template<class Index, class Function, class Shape>
__AGENCY_ANNOTATION
project_index_and_invoke<Index,Function,Shape>
  make_project_index_and_invoke(Function f,
                                Shape higher_dimensional_shape,
                                typename project_index_and_invoke<Index,Function,Shape>::projected_shape_type lower_dimensional_shape)
{
  return project_index_and_invoke<Index,Function,Shape>{f,higher_dimensional_shape,lower_dimensional_shape};
}


} // end detail
} // end agency

