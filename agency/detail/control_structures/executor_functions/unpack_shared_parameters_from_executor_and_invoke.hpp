#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/control_structures/executor_functions/shared_parameter_packaging.hpp>
#include <agency/detail/tuple.hpp>

namespace agency
{
namespace detail
{


// this is the functor that bulk functions like bulk_invoke
// execute via an executor to unpack potentially many shared parameters
// from the packaging used with the executor and the invoke the given function
// with the execution agent index and unpacked shared parameters as arguments
template<class Function>
struct unpack_shared_parameters_from_executor_and_invoke
{
  mutable Function g;

  template<class Index, class... Types>
  __AGENCY_ANNOTATION
  auto operator()(const Index& idx, Types&... packaged_shared_params) const
    -> decltype(
         __tu::tuple_apply(
           g,
           __tu::tuple_prepend_invoke(
             agency::detail::unpack_shared_parameters_from_executor(packaged_shared_params...),
             idx,
             agency::detail::forwarder{})
         )
       )
  {
    auto shared_params = agency::detail::unpack_shared_parameters_from_executor(packaged_shared_params...);

    // XXX the following is the moral equivalent of:
    // g(idx, shared_params...);

    // create one big tuple of the arguments so we can just call tuple_apply
    auto idx_and_shared_params = __tu::tuple_prepend_invoke(shared_params, idx, agency::detail::forwarder{});

    return __tu::tuple_apply(g, idx_and_shared_params);
  }
};

template<class Function>
__AGENCY_ANNOTATION
unpack_shared_parameters_from_executor_and_invoke<Function> make_unpack_shared_parameters_from_executor_and_invoke(Function f)
{
  return unpack_shared_parameters_from_executor_and_invoke<Function>{f};
}


} // end detail
} // end agency

