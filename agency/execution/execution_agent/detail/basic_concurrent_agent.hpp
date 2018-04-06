#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/executor/properties/bulk_guarantee.hpp>
#include <agency/execution/execution_agent/detail/basic_execution_agent.hpp>
#include <agency/detail/concurrency/barrier.hpp>
#include <agency/detail/concurrency/in_place_barrier.hpp>
#include <agency/container/array.hpp>
#include <agency/experimental/optional.hpp>
#include <agency/experimental/variant.hpp>
#include <type_traits>


namespace agency
{
namespace detail
{


template<class Index, class Barrier, class MemoryResource>
class basic_concurrent_agent : public detail::basic_execution_agent<bulk_guarantee_t::concurrent_t, Index>
{
  private:
    using super_t = detail::basic_execution_agent<bulk_guarantee_t::concurrent_t, Index>;

    // wrap the user's Barrier type with in_place_barrier so that we may use
    // in_place_type_t with its constructor
    using barrier_type = in_place_barrier<Barrier>;

    static constexpr size_t broadcast_channel_size = sizeof(void*);
    using broadcast_channel_type = agency::array<char, broadcast_channel_size>;

    // this function destroys *ptr if ptr is not null and then makes the entire group wait
    // only one agent should pass a non-nullptr to this function
    // the entire group should be convergent before calling this function
    template<class T>
    __AGENCY_ANNOTATION
    typename std::enable_if<
      std::is_trivially_destructible<T>::value
    >::type
      destroy_and_wait_if(T*)
    {
      // no op: T has a trivial destructor, so there's no need to do anything
      // including synchronize
    }

    // this function destroys *ptr if ptr is not null and then makes the entire group wait
    // only one agent should pass a non-nullptr to this function
    // the entire group should be convergent before calling this function
    template<class T>
    __AGENCY_ANNOTATION
    typename std::enable_if<
      !std::is_trivially_destructible<T>::value
    >::type
      destroy_and_wait_if(T* ptr)
    {
      // first destroy the object
      if(ptr)
      {
        ptr->~T();
      }

      // synchronize the group
      wait();
    }

    
    // this overload of broadcast_impl() is for small T
    template<class T,
             __AGENCY_REQUIRES(
               (sizeof(T) <= broadcast_channel_size)
             )>
    __AGENCY_ANNOTATION
    T broadcast_impl(const experimental::optional<T>& value)
    {
      // value is small enough to fit inside broadcast_channel_, so we can
      // send it through directly without needing to dynamically allocate storage
      
      // reinterpret the broadcast channel into the right kind of type
      T* shared_temporary_object = reinterpret_cast<T*>(shared_param_.broadcast_channel_.data());

      // the thread with the value copies it into a shared temporary
      if(value)
      {
        // copy construct the shared temporary
        ::new(shared_temporary_object) T(*value);
      }

      // all agents wait for the object to be ready
      wait();

      // copy the shared temporary to a local variable
      T result = *shared_temporary_object;

      // all agents wait for all other agents to finish copying the shared temporary
      wait();

      // destroy the object and all agents wait for the broadcast to become ready again
      destroy_and_wait_if(value ? shared_temporary_object : nullptr);

      return result;
    }


    // this overload of broadcast_impl() is for large T
    template<class T,
             __AGENCY_REQUIRES(
               (sizeof(T) > broadcast_channel_size)
             )>
    __AGENCY_ANNOTATION
    T broadcast_impl(const experimental::optional<T>& value)
    {
      // value is too large to fit through broadcast_channel_, so
      // we need to dynamically allocate storage

      // reinterpret the broadcast channel into a pointer
      static_assert(sizeof(broadcast_channel_type) >= sizeof(T*), "broadcast channel is too small to accomodate T*");
      T* shared_temporary_object = reinterpret_cast<T*>(&shared_param_.broadcast_channel_);

      if(value)
      {
        // dynamically allocate the shared temporary object
        shared_temporary_object = memory_resource().allocate<alignof(T)>(sizeof(T));

        // copy construct the shared temporary
        ::new(shared_temporary_object) T(*value);
      }

      // all agents wait for the object to be ready
      wait();

      // copy the shared temporary to a local variable
      T result = *shared_temporary_object;

      // all agents wait for other agents to finish copying the shared temporary
      wait();

      if(value)
      {
        // destroy the shared temporary
        shared_temporary_object->~T();

        // deallocate the temporary storage
        memory_resource().deallocate(shared_temporary_object);
      }

      // all agents wait for the broadcast channel and memory resource to become ready again
      wait();

      return result;
    }

  public:
    using param_type = typename super_t::param_type;

    using index_type = typename super_t::index_type;

    __AGENCY_ANNOTATION
    void wait() const
    {
      shared_param_.barrier_.arrive_and_wait();
    }

    template<class T>
    __AGENCY_ANNOTATION
    T broadcast(const experimental::optional<T>& value)
    {
      return broadcast_impl(value);
    }

    using memory_resource_type = MemoryResource;

    __AGENCY_ANNOTATION
    memory_resource_type& memory_resource()
    {
      return shared_param_.memory_resource_;
    }

    class shared_param_type
    {
      public:
        __AGENCY_ANNOTATION
        shared_param_type(const param_type& param)
          : barrier_(param.domain().size()),
            memory_resource_()
        {
          // note we specifically avoid default constructing broadcast_channel_
        }

        template<class OtherBarrier,
                 __AGENCY_REQUIRES(
                   std::is_constructible<barrier_type, experimental::in_place_type_t<OtherBarrier>, std::size_t>::value
                )>
        __AGENCY_ANNOTATION
        shared_param_type(const param_type& param, experimental::in_place_type_t<OtherBarrier> which_barrier)
          : barrier_(which_barrier, param.domain().size()),
            memory_resource_()
        {
          // note we specifically avoid default constructing broadcast_channel_
        }

        // shared_param_type needs to be moveable, even if its member types aren't,
        // because shared_param_type objects will be returned from factory functions
        // at the moment, this requires moveability
        //
        // XXX we should be able to eliminate this move constructor in C++17
        //     see wg21.link/P0135
        __AGENCY_ANNOTATION
        shared_param_type(shared_param_type&& other)
          : barrier_(other.barrier_.index(), other.barrier_.count()),
            memory_resource_()
        {}

      private:
        // broadcast_channel_ needs to be the first member to ensure proper alignment because we reinterpret it to arbitrary T*
        // XXX is there a more comprehensive way to ensure that this member falls on the right address?
        broadcast_channel_type broadcast_channel_;
        barrier_type barrier_;
        memory_resource_type memory_resource_;

        friend basic_concurrent_agent;
    };

  private:
    shared_param_type& shared_param_;

  protected:
    __AGENCY_ANNOTATION
    basic_concurrent_agent(const index_type& index, const param_type& param, shared_param_type& shared_param)
      : super_t(index, param),
        shared_param_(shared_param)
    {}

    // friend execution_agent_traits to give it access to the constructor
    friend struct agency::execution_agent_traits<basic_concurrent_agent>;
};


} // end detail
} // end agency

