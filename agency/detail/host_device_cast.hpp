#pragma once

#include <agency/detail/config.hpp>
#include <utility>

namespace agency
{
namespace detail
{


template<class FunctionReference>
struct host_device_function
{
  FunctionReference f_;

  // XXX this should use agency::invoke(), but there's a circular dependency
  // between this header and functional.hpp
  __agency_hd_warning_disable__
  template<class... Args>
  __AGENCY_ANNOTATION
  auto operator()(Args&&... args) const
    -> decltype(
         std::forward<FunctionReference>(f_)(std::forward<Args>(args)...)
       )
  {
    return std::forward<FunctionReference>(f_)(std::forward<Args>(args)...);
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

