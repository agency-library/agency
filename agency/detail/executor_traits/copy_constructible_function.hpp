#pragma once

#include <agency/detail/config.hpp>
#include <agency/functional.hpp>
#include <utility>
#include <memory>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class Function>
struct shared_function
{
  std::shared_ptr<Function> f_ptr;

  template<class Function1,
           class = typename std::enable_if<
             std::is_constructible<
               Function,Function1&&
             >::value
           >::type>
  shared_function(Function1&& f)
    : f_ptr(std::make_shared<Function>(std::forward<Function1>(f)))
  {}

  template<class... Args>
  auto operator()(Args&&... args) ->
    decltype(agency::invoke(*f_ptr, std::forward<Args>(args)...))
  {
    return agency::invoke(*f_ptr, std::forward<Args>(args)...);
  }

  template<class... Args>
  auto operator()(Args&&... args) const ->
    decltype(agency::invoke(*f_ptr, std::forward<Args>(args)...))
  {
    return agency::invoke(*f_ptr, std::forward<Args>(args)...);
  }
};


template<class Function>
struct copy_constructible_function
{
  using type = typename std::conditional<
    std::is_copy_constructible<Function>::value,
    Function,
    shared_function<Function>
  >::type;
};


template<class Function>
using copy_constructible_function_t = typename copy_constructible_function<Function>::type;


} // end new_executor_traits_detail
} // end detail
} // end agency

