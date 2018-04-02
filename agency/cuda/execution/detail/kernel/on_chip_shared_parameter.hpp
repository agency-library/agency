#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/factory.hpp>
#include <agency/detail/uninitialized.hpp>
#include <agency/detail/tuple/tuple_utility.hpp>
#include <type_traits>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class Factory>
struct result_of_factory_is_empty
  : std::integral_constant<
      bool,
      (std::is_empty<agency::detail::result_of_t<Factory()>>::value ||
      agency::detail::is_empty_tuple<agency::detail::result_of_t<Factory()>>::value)
    >
{};


template<class Factory, bool = result_of_factory_is_empty<Factory>::value>
struct on_chip_shared_parameter
{
  using value_type = agency::detail::result_of_t<Factory()>;

  inline __device__
  on_chip_shared_parameter(bool is_leader, Factory factory)
    : is_leader_(is_leader)
  {
    __shared__ agency::detail::uninitialized<value_type> inner_shared_param;

    if(is_leader_)
    {
      inner_shared_param.construct(factory());
    }

#ifdef __CUDA_ARCH__
    __syncthreads();
#endif

    inner_shared_param_ = &inner_shared_param;
  }

  on_chip_shared_parameter(const on_chip_shared_parameter&) = delete;
  on_chip_shared_parameter(on_chip_shared_parameter&&) = delete;

  inline __device__
  ~on_chip_shared_parameter()
  {
#ifdef __CUDA_ARCH__
    __syncthreads();
#endif

    if(is_leader_)
    {
      inner_shared_param_->destroy();
    }
  }

  inline __device__
  value_type& get()
  {
    return inner_shared_param_->get();
  }

  const bool is_leader_;
  agency::detail::uninitialized<value_type>* inner_shared_param_;
};


template<class Factory>
struct on_chip_shared_parameter<Factory,true>
{
  using value_type = agency::detail::result_of_t<Factory()>;

  inline __device__
  on_chip_shared_parameter(bool is_leader_, Factory) {}

  on_chip_shared_parameter(const on_chip_shared_parameter&) = delete;
  on_chip_shared_parameter(on_chip_shared_parameter&&) = delete;

  inline __device__
  value_type& get()
  {
    return param_;
  }

  value_type param_;
};


} // end detail
} // end cuda
} // end agency

