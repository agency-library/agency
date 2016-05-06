#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/invoke.hpp>
#include <utility>

namespace agency
{
namespace detail
{


template<class FunctionReference>
struct host_device_function
{
  FunctionReference f_;

  __agency_exec_check_disable__
  template<class... Args>
  __AGENCY_ANNOTATION
  auto operator()(Args&&... args) const
    -> decltype(
         agency::detail::invoke(std::forward<FunctionReference>(f_), std::forward<Args>(args)...)
       )
  {
    return agency::detail::invoke(std::forward<FunctionReference>(f_), std::forward<Args>(args)...);
  }
};


template<class Function>
__AGENCY_ANNOTATION
host_device_function<Function&&> host_device_cast(Function&& f)
{
  return host_device_function<Function&&>{std::forward<Function>(f)};
}


// avoid nested wrapping
template<class FunctionReference>
__AGENCY_ANNOTATION
host_device_function<FunctionReference> host_device_cast(host_device_function<FunctionReference> f)
{
  return f;
}


} // end detail
} // end agency

