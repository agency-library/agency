#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/concurrency/barrier.hpp>
#include <agency/cuda/detail/concurrency/block_barrier.hpp>
#include <agency/cuda/detail/concurrency/grid_barrier.hpp>
#include <agency/cuda/detail/concurrency/heterogeneous_barrier.hpp>
#include <agency/detail/concurrency/variant_barrier.hpp>
#include <agency/experimental/variant.hpp>
#include <cstddef>
#include <type_traits>


namespace agency
{
namespace detail
{


// any_barrier is a type eraser for barrier types
// At the moment, its functionality is limited to the types of barriers present within Agency
// In other words, there is currently no support for user-defined barriers
class any_barrier
{
  private:
    // we subclass heterogeneous_barrier so that we can introduce a type whose interface matches variant_barrier
    // because the two types are used uniformly below
    class host_or_device_barrier : public cuda::detail::heterogeneous_barrier<barrier, cuda::detail::block_barrier>
    {
      private:
        using super_t = cuda::detail::heterogeneous_barrier<barrier, cuda::detail::block_barrier>;

      public:
        __AGENCY_ANNOTATION
        host_or_device_barrier(std::size_t count)
          : super_t(count)
        {}

        // give this type constructors which match variant_barrier's in-place constructors
        // these constructors simply ignore the additional argument
        __AGENCY_ANNOTATION
        host_or_device_barrier(experimental::in_place_type_t<detail::barrier>, std::size_t count)
          : host_or_device_barrier(count)
        {}

        __AGENCY_ANNOTATION
        host_or_device_barrier(experimental::in_place_type_t<cuda::detail::block_barrier>, std::size_t count)
          : host_or_device_barrier(count)
        {}

        __AGENCY_ANNOTATION
        host_or_device_barrier(std::size_t /*index*/, std::size_t count)
          : host_or_device_barrier(count)
        {}

        __AGENCY_ANNOTATION
        std::size_t index() const
        {
          return 0;
        }
    };

    // Depending on the availability of cooperative_groups, the implementation of any_barrier is either
    // 
    //   1. A variant_barrier (which modulates its behavior based on its dynamic runtime state), or
    //   2. A host_or_device_barrier (which modulates its behavior based simply on whether its functions are called from __host__ or __device__ code)
    //
    // The reason we do this is to avoid tracking the type of the barrier stored inside any_barrier at runtime in order to
    // save a little bit of storage.
    using implementation_type =
#if __cuda_lib_has_cooperative_groups
      detail::variant_barrier<
        detail::barrier,
        cuda::detail::block_barrier,
        cuda::detail::grid_barrier
      >
#else
      host_or_device_barrier
#endif
    ;

    implementation_type implementation_;

  public:
    template<class Barrier,
             __AGENCY_REQUIRES(
               std::is_constructible<implementation_type, experimental::in_place_type_t<Barrier>, std::size_t>::value
             )>
    __AGENCY_ANNOTATION
    any_barrier(experimental::in_place_type_t<Barrier> which_barrier, std::size_t count)
      : implementation_(which_barrier, count)
    {}

#if !__cuda_lib_has_cooperative_groups
    // if the user asks for a grid_barrier, and cooperative_groups is not available,
    // emit an error
    // use a deduced template non-type parameter here so that the static_assert below is
    // raised only if this constructor is called
    template<bool deduced_false = false>
    __AGENCY_ANNOTATION
    any_barrier(experimental::in_place_type_t<cuda::detail::grid_barrier>, std::size_t count)
      : any_barrier(count)
    {
      static_assert(deduced_false, "any_barrier constructor: Constructing a cuda::detail::grid_barrier requires CUDA version >= 9, __CUDA_ARCH__ >= 600, and relocatable device code.");
    }
#endif

    __AGENCY_ANNOTATION
    any_barrier(std::size_t index, std::size_t count)
      : implementation_(index, count)
    {}

    // by default, any_barrier constructs itself as an agency::detail::barrier
    __AGENCY_ANNOTATION
    any_barrier(std::size_t count)
      : any_barrier(experimental::in_place_type_t<agency::detail::barrier>(), count)
    {}

    __AGENCY_ANNOTATION
    std::size_t count() const
    {
      return implementation_.count();
    }

    __AGENCY_ANNOTATION
    void arrive_and_wait()
    {
      implementation_.arrive_and_wait();
    }

    // XXX this function should be eliminated when shared_param_type's move constructor is eliminated pending C++17
    //     see wg21.link/P0135
    __AGENCY_ANNOTATION
    std::size_t index() const
    {
      return implementation_.index();
    }
};


} // end detail
} // end agency

