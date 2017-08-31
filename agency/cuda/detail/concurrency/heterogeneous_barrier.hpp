#include <agency/detail/config.hpp>


namespace agency
{
namespace cuda
{
namespace detail
{


template<class HostBarrier, class DeviceBarrier>
class heterogeneous_barrier
{
  public:
    using host_barrier_type = HostBarrier;
    using device_barrier_type = DeviceBarrier;

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    heterogeneous_barrier(size_t num_threads) :
#ifndef __CUDA_ARCH__
      host_barrier_(num_threads)
#else
      device_barrier_(num_threads)
#endif
    {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    std::size_t count() const
    {
#ifndef __CUDA_ARCH__
       return host_barrier_.count();
#else
       return device_barrier_.count();
#endif
    }

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    void arrive_and_wait()
    {
#ifndef __CUDA_ARCH__
      host_barrier_.arrive_and_wait();
#else
      device_barrier_.arrive_and_wait();
#endif
    }

  private:
#ifndef __CUDA_ARCH__
    host_barrier_type host_barrier_;
#else
    device_barrier_type device_barrier_;
#endif
};


} // end detail
} // end cuda
} // end agency

