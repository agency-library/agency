#pragma once

#include <agency/execution/execution_categories.hpp>
#include <agency/detail/concurrency/barrier.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/index_tuple.hpp>
#include <agency/detail/index_cast.hpp>
#include <agency/detail/unwrap_tuple_if_not_scoped.hpp>
#include <agency/detail/make_tuple_if_not_scoped.hpp>
#include <agency/detail/memory/arena_resource.hpp>
#include <agency/coordinate.hpp>
#include <agency/experimental/array.hpp>
#include <agency/experimental/optional.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{


__DEFINE_HAS_MEMBER_TYPE(has_param_type, param_type);
__DEFINE_HAS_MEMBER_TYPE(has_shared_param_type, shared_param_type);
__DEFINE_HAS_MEMBER_TYPE(has_inner_execution_agent_type, inner_execution_agent_type);


template<class ExecutionAgent, class Enable = void>
struct execution_agent_traits_base
{
};


template<class ExecutionAgent>
struct execution_agent_traits_base<
  ExecutionAgent,
  typename std::enable_if<
    has_inner_execution_agent_type<ExecutionAgent>::type
  >::type
>
{
  using inner_execution_agent_type = typename ExecutionAgent::inner_execution_agent_type;
};


template<class ExecutionAgent, class Enable = void>
struct execution_agent_type_list
{
  using type = type_list<ExecutionAgent>;
};

template<class ExecutionAgent>
struct execution_agent_type_list<
  ExecutionAgent,
  typename std::enable_if<
    has_inner_execution_agent_type<ExecutionAgent>::value
  >::type
>
{
  using type = typename type_list_prepend<
    ExecutionAgent,
    typename execution_agent_type_list<
      typename ExecutionAgent::inner_execution_agent_type
    >::type
  >::type;
};


// derive from ExecutionAgent to get access to ExecutionAgent's constructor
template<class ExecutionAgent>
struct agent_access_helper : public ExecutionAgent
{
  template<class... Args>
  __AGENCY_ANNOTATION
  agent_access_helper(Args&&... args)
    : ExecutionAgent(std::forward<Args>(args)...)
  {}
};


// make_agent() is a helper function used by execution_agent_traits and execution_group. its job is to simplify the job of creating an
// execution agent by calling its constructor
// make_agent() forwards index & param and filters out ignored shared parameters when necessary
// in other words, when shared_param is ignore_t, it doesn't pass shared_param to the agent's constructor
// in other cases, it forwards along the shared_param
template<class ExecutionAgent, class Index, class Param>
__AGENCY_ANNOTATION
ExecutionAgent make_agent(const Index& index,
                          const Param& param)
{
  return agent_access_helper<ExecutionAgent>(index, param);
}


template<class ExecutionAgent, class Index, class Param>
__AGENCY_ANNOTATION
static ExecutionAgent make_flat_agent(const Index& index,
                                      const Param& param,
                                      agency::detail::ignore_t)
{
  return make_agent<ExecutionAgent>(index, param);
}


template<class ExecutionAgent, class Index, class Param, class SharedParam>
__AGENCY_ANNOTATION
static ExecutionAgent make_flat_agent(const Index& index,
                                      const Param& param,
                                      SharedParam& shared_param)
{
  return agent_access_helper<ExecutionAgent>(index, param, shared_param);
}


template<class ExecutionAgent, class Index, class Param, class SharedParam>
__AGENCY_ANNOTATION
static ExecutionAgent make_agent(const Index& index,
                                 const Param& param,
                                 SharedParam& shared_param)
{
  return make_flat_agent<ExecutionAgent>(index, param, shared_param);
}


template<class ExecutionAgent, class Index, class Param, class SharedParam1, class SharedParam2, class... SharedParams>
__AGENCY_ANNOTATION
static ExecutionAgent make_agent(const Index& index,
                                 const Param& param,
                                 SharedParam1&    shared_param1,
                                 SharedParam2&    shared_param2,
                                 SharedParams&... shared_params)
{
  return agent_access_helper<ExecutionAgent>(index, param, shared_param1, shared_param2, shared_params...);
}


} // end detail


template<class ExecutionAgent>
struct execution_agent_traits : detail::execution_agent_traits_base<ExecutionAgent>
{
  using execution_agent_type = ExecutionAgent;
  using execution_category = typename execution_agent_type::execution_category;

  // XXX we should probably use execution_agent_type::index_type if it exists,
  //     if not, use the type of the result of .index()
  // XXX WAR cudafe performance issue
  //using index_type = detail::decay_t<
  //  decltype(
  //    std::declval<execution_agent_type>().index()
  //  )
  //>;
  using index_type = typename execution_agent_type::index_type;

  using size_type = detail::decay_t<
    decltype(
      std::declval<execution_agent_type>().group_size()
    )
  >;

  private:
    template<class T>
    struct execution_agent_param
    {
      using type = typename T::param_type;
    };

  public:

    using param_type = typename detail::lazy_conditional<
      detail::has_param_type<execution_agent_type>::value,
      execution_agent_param<execution_agent_type>,
      detail::identity<size_type>
    >::type;

    // XXX what should we do if ExecutionAgent::domain(param) does not exist?
    //     default should be lattice<index_type>, but by what process should we eventually
    //     arrive at that default?
    // XXX yank the general implementation from execution_group now that param_type::inner() exists
    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    static auto domain(const param_type& param)
      -> decltype(ExecutionAgent::domain(param))
    {
      return ExecutionAgent::domain(param);
    }

    using domain_type = decltype(domain(std::declval<param_type>()));

    template<class Function>
    __AGENCY_ANNOTATION
    static detail::result_of_t<Function(ExecutionAgent&)>
      execute(Function f, const index_type& index, const param_type& param)
    {
      ExecutionAgent agent(index, param);
      return f(agent);
    }


  private:
    template<class T>
    struct execution_agent_shared_param
    {
      using type = typename T::shared_param_type;
    };

  public:
    using shared_param_type = typename detail::lazy_conditional<
      detail::has_shared_param_type<execution_agent_type>::value,
      execution_agent_shared_param<execution_agent_type>,
      detail::identity<agency::detail::ignore_t>
    >::type;

    // XXX we should ensure that the SharedParams are all the right type for each inner execution agent type
    //     basically, they would be the element types of shared_param_tuple_type
    template<class Function, class SharedParam1, class... SharedParams>
    __AGENCY_ANNOTATION
    static detail::result_of_t<Function(ExecutionAgent&)>
      execute(Function f, const index_type& index, const param_type& param, SharedParam1& shared_param1, SharedParams&... shared_params)
    {
      ExecutionAgent agent = detail::make_agent<ExecutionAgent>(index, param, shared_param1, shared_params...);
      return f(agent);
    }
};


namespace detail
{


template<class ExecutionCategory, class Index = size_t>
class basic_execution_agent
{
  public:
    using execution_category = ExecutionCategory;

    using index_type = Index;

    __AGENCY_ANNOTATION
    index_type index() const
    {
      return index_;
    }

    using domain_type = lattice<index_type>;

    __AGENCY_ANNOTATION
    const domain_type& domain() const
    {
      return domain_;
    }

    using size_type = decltype(std::declval<domain_type>().size());

    __AGENCY_ANNOTATION
    size_type group_size() const
    {
      return domain().size();
    }

    __AGENCY_ANNOTATION
    auto group_shape() const
      -> decltype(this->domain().shape())
    {
      return domain().shape();
    }

    __AGENCY_ANNOTATION
    size_type rank() const
    {
      return index_cast<size_type>(index(), group_shape(), group_size());
    }

    __AGENCY_ANNOTATION
    bool elect() const
    {
      return rank() == 0;
    }

    class param_type
    {
      public:
        __AGENCY_ANNOTATION
        param_type() = default;

        __AGENCY_ANNOTATION
        param_type(const param_type& other) = default;

        __AGENCY_ANNOTATION
        param_type(const domain_type& d)
          : domain_(d)
        {}

        __AGENCY_ANNOTATION
        param_type(const index_type& min, const index_type& max)
          : param_type(domain_type(min,max))
        {}

        __AGENCY_ANNOTATION
        const domain_type& domain() const
        {
          return domain_;
        }

      private:
        domain_type domain_;
    };

    __AGENCY_ANNOTATION
    static domain_type domain(const param_type& p)
    {
      return p.domain();
    }


  protected:
    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    basic_execution_agent(const index_type& index, const param_type& param) : index_(index), domain_(param.domain()) {}

    friend struct agency::execution_agent_traits<basic_execution_agent>;

  private:
    index_type index_;
    domain_type domain_;
};


} // end detail


using sequenced_agent = detail::basic_execution_agent<sequenced_execution_tag>;
using sequenced_agent_1d = sequenced_agent;
using sequenced_agent_2d = detail::basic_execution_agent<sequenced_execution_tag, size2>;


using parallel_agent = detail::basic_execution_agent<parallel_execution_tag>;
using parallel_agent_1d = parallel_agent;
using parallel_agent_2d = detail::basic_execution_agent<parallel_execution_tag, size2>;


using unsequenced_agent = detail::basic_execution_agent<unsequenced_execution_tag>;
using unsequenced_agent_1d = unsequenced_agent;
using unsequenced_agent_2d = detail::basic_execution_agent<unsequenced_execution_tag, size2>;


namespace detail
{



template<class Index, class MemoryResource = detail::arena_resource<128 * sizeof(int)>>
class basic_concurrent_agent : public detail::basic_execution_agent<concurrent_execution_tag, Index>
{
  private:
    using super_t = detail::basic_execution_agent<concurrent_execution_tag, Index>;

    static constexpr size_t broadcast_channel_size = sizeof(void*);
    using broadcast_channel_type = agency::experimental::array<char, broadcast_channel_size>;

    // this class hides agency::detail::barrier & __syncthreads()
    // behind a uniform interface so that we can use basic_concurrent_agent
    // in both C++ and CUDA C++
    class barrier
    {
      public:
        __AGENCY_ANNOTATION
        barrier(size_t num_threads)
#ifndef __CUDA_ARCH__
         : barrier_(num_threads)
#endif
        {}

        __AGENCY_ANNOTATION
        void arrive_and_wait()
        {
#ifndef __CUDA_ARCH__
          barrier_.arrive_and_wait();
#else
          __syncthreads();
#endif
        }

#ifndef __CUDA_ARCH__
      private:
        agency::detail::barrier barrier_;
#endif
    };

    template<class T>
    __AGENCY_ANNOTATION
    typename std::enable_if<
      std::is_trivially_destructible<T>::value
    >::type
      wait_and_destroy_if(T*, bool)
    {
      // no op: T has a trivial destructor, so there's no need to synchronize
      // and call T's destructor
    }

    template<class T>
    __AGENCY_ANNOTATION
    typename std::enable_if<
      !std::is_trivially_destructible<T>::value
    >::type
      wait_and_destroy_if(T* ptr, bool only_this_agent_should_call_the_destructor)
    {
      // first, synchronize
      wait();

      if(only_this_agent_should_call_the_destructor)
      {
        ptr->~T();
      }
    }


    template<class T>
    using enable_if_small_enough_to_broadcast_directly_t = typename std::enable_if<
      sizeof(T) <= broadcast_channel_size, T
    >::type;

    // this overload of broadcast_impl() is for small T
    template<class T>
    __AGENCY_ANNOTATION
    enable_if_small_enough_to_broadcast_directly_t<T>
      broadcast_impl(const experimental::optional<T>& value)
    {
      // value is small enough to fit inside broadcast_channel_, so we can
      // send it through directly without needing to dynamically allocating storage
      
      // reinterpret the broadcast channel into the right kind of type
      T* shared_temporary_object = reinterpret_cast<T*>(broadcast_channel_.data());

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

      // synchronize and destroy the object if necessary
      // XXX do we need to unconditionally wait here?
      //     what if a subsequent call to broadcast() happens?
      //     will the subsequent call collide with this one?
      // XXX should we just decline to synchronize the group before returning?
      wait_and_destroy_if(shared_temporary_object, bool(value));

      return result;
    }


    template<class T>
    using enable_if_too_large_to_broadcast_directly_t = typename std::enable_if<
      (sizeof(T) > broadcast_channel_size), T
    >::type;

    // this overload of broadcast_impl() is for large T
    template<class T>
    __AGENCY_ANNOTATION
    enable_if_too_large_to_broadcast_directly_t<T>
      broadcast_impl(const experimental::optional<T>& value)
    {
      // value is too large to fit through broadcast_channel_, so
      // we need to dynamically allocate storage

      // reinterpret the broadcast channel into a pointer
      static_assert(sizeof(broadcast_channel_) >= sizeof(T*), "broadcast channel is too small to accomodate T*");
      T* shared_temporary_object = reinterpret_cast<T*>(&broadcast_channel_);

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
    __AGENCY_ANNOTATION
    void wait() const
    {
      barrier_.arrive_and_wait();
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
      return memory_resource_;
    }

    struct shared_param_type
    {
      __AGENCY_ANNOTATION
      shared_param_type(const typename super_t::param_type& param)
        : barrier_(param.domain().size()),
          memory_resource_(),
          count_(param.domain().size())
      {
        // note we specifically avoid default constructing broadcast_channel_
      }

      // XXX see if we can eliminate this copy constructor
      //     i'm not certain it's necessary to copy shared_param_type anymore
      __AGENCY_ANNOTATION
      shared_param_type(const shared_param_type& other)
        : barrier_(other.count_),
          memory_resource_(),
          count_(other.count_)
      {}

      // broadcast_channel_ needs to be the first member to ensure proper alignment because we reinterpret it to arbitrary T*
      // XXX is there a more comprehensive way to ensure that this member falls on the right address?
      broadcast_channel_type broadcast_channel_;
      barrier barrier_;
      memory_resource_type memory_resource_;
      int count_;
    };

  private:
    barrier& barrier_;
    broadcast_channel_type& broadcast_channel_;
    memory_resource_type& memory_resource_;

  protected:
    __AGENCY_ANNOTATION
    basic_concurrent_agent(const typename super_t::index_type& index, const typename super_t::param_type& param, shared_param_type& shared_param)
      : super_t(index, param),
        barrier_(shared_param.barrier_),
        broadcast_channel_(shared_param.broadcast_channel_),
        memory_resource_(shared_param.memory_resource_)
    {}

    // friend execution_agent_traits to give it access to the constructor
    friend struct agency::execution_agent_traits<basic_concurrent_agent>;
};


} // end detail


using concurrent_agent = detail::basic_concurrent_agent<size_t>;
using concurrent_agent_1d = concurrent_agent;
using concurrent_agent_2d = detail::basic_concurrent_agent<size2>;


namespace detail
{


template<class OuterExecutionAgent, class Enable = void>
struct execution_group_base {};


// if execution_group's OuterExecutionAgent has a shared_param_type,
// then execution_group needs to have a shared_param_type which can be constructed from our param_type
template<class OuterExecutionAgent>
struct execution_group_base<OuterExecutionAgent,
                            typename std::enable_if<
                              detail::has_shared_param_type<OuterExecutionAgent>::value
                            >::type>
{
  struct shared_param_type : public OuterExecutionAgent::shared_param_type
  {
    template<class ParamType>
    __AGENCY_ANNOTATION
    shared_param_type(const ParamType& param) : OuterExecutionAgent::shared_param_type(param.outer()) {}
  };
};


template<class OuterExecutionAgent, class InnerExecutionAgent>
class execution_group : public execution_group_base<OuterExecutionAgent>
{
  private:
    using outer_traits = execution_agent_traits<OuterExecutionAgent>;
    using inner_traits = execution_agent_traits<InnerExecutionAgent>;

    using outer_execution_category = typename outer_traits::execution_category;
    using inner_execution_category = typename inner_traits::execution_category;

    using outer_index_type = typename outer_traits::index_type;
    using inner_index_type = typename inner_traits::index_type;

  public:
    using index_type = decltype(
      __tu::tuple_cat_apply(
        agency::detail::index_tuple_maker{},
        agency::detail::make_tuple_if_not_scoped<outer_execution_category>(std::declval<outer_index_type>()),
        agency::detail::make_tuple_if_not_scoped<inner_execution_category>(std::declval<inner_index_type>())
      )
    );

  private:
    // concatenates an outer index with an inner index
    // returns an index_tuple with arithmetic ops (not a std::tuple)
    // XXX move this into index_tuple.hpp?
    __AGENCY_ANNOTATION
    static index_type index_cat(const outer_index_type& outer_idx, const inner_index_type& inner_idx)
    {
      return __tu::tuple_cat_apply(
        agency::detail::index_tuple_maker{},
        agency::detail::make_tuple_if_not_scoped<outer_execution_category>(outer_idx),
        agency::detail::make_tuple_if_not_scoped<inner_execution_category>(inner_idx)
      );
    }

  public:
    using execution_category = scoped_execution_tag<
      outer_execution_category,
      inner_execution_category
    >;

    using outer_execution_agent_type = OuterExecutionAgent;
    using inner_execution_agent_type = InnerExecutionAgent;

    class param_type
    {
      private:
        typename outer_traits::param_type outer_;
        typename inner_traits::param_type inner_;

      public:
        __AGENCY_ANNOTATION
        param_type() = default;

        __AGENCY_ANNOTATION
        param_type(const param_type&) = default;

        __AGENCY_ANNOTATION
        param_type(const typename outer_traits::param_type& o, const typename inner_traits::param_type& i) : outer_(o), inner_(i) {}

        __AGENCY_ANNOTATION
        const typename outer_traits::param_type& outer() const
        {
          return outer_;
        }

        __AGENCY_ANNOTATION
        const typename inner_traits::param_type& inner() const
        {
          return inner_;
        }
    };

    __AGENCY_ANNOTATION
    outer_execution_agent_type& outer()
    {
      return outer_agent_;
    }

    __AGENCY_ANNOTATION
    const outer_execution_agent_type& outer() const
    {
      return outer_agent_;
    }

    __AGENCY_ANNOTATION
    inner_execution_agent_type& inner()
    {
      return inner_agent_;
    }

    __AGENCY_ANNOTATION
    const inner_execution_agent_type& inner() const
    {
      return inner_agent_;
    }

    __AGENCY_ANNOTATION
    index_type index() const
    {
      return index_cat(this->outer().index(), this->inner().index());
    }

    using domain_type = lattice<index_type>;

    __AGENCY_ANNOTATION
    domain_type domain() const
    {
      auto outer_domain = outer().domain();
      auto inner_domain = this->inner().domain();

      auto min = index_cat(outer_domain.min(), inner_domain.min());
      auto max = index_cat(outer_domain.max(), inner_domain.max());

      return domain_type{min,max};
    }

    // XXX can probably move this to execution_agent_traits
    __AGENCY_ANNOTATION
    static domain_type domain(const param_type& param)
    {
      auto outer_domain = outer_traits::domain(param.outer());
      auto inner_domain = inner_traits::domain(param.inner());

      auto min = index_cat(outer_domain.min(), inner_domain.min());
      auto max = index_cat(outer_domain.max(), inner_domain.max());

      return domain_type{min,max};
    }
    
    __AGENCY_ANNOTATION
    auto group_shape() const
      -> decltype(this->domain().shape())
    {
      return domain().shape();
    }

    __AGENCY_ANNOTATION
    auto group_size() const
      -> decltype(this->outer().group_size() * inner().group_size())
    {
      return outer().group_size() * inner().group_size();
    }

    __AGENCY_ANNOTATION
    auto rank() const
      -> decltype(this->group_size())
    {
      using size_type = decltype(this->group_size());
      return index_cast<size_type>(index(), group_shape(), group_size());
    }

    __AGENCY_ANNOTATION
    bool elect() const
    {
      return outer().elect() && inner().elect();
    }

  protected:
    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    execution_group(const index_type& index, const param_type& param)
      : outer_agent_(detail::make_agent<outer_execution_agent_type>(outer_index(index), param.outer())),
        inner_agent_(detail::make_agent<inner_execution_agent_type>(inner_index(index), param.inner()))
    {}

    // XXX ensure all the shared params are the right type
    __agency_exec_check_disable__
    template<class SharedParam1, class... SharedParams>
    __AGENCY_ANNOTATION
    execution_group(const index_type& index, const param_type& param, SharedParam1& shared_param1, SharedParams&... shared_params)
      : outer_agent_(agency::detail::make_agent<outer_execution_agent_type>(outer_index(index), param.outer(), shared_param1)),
        inner_agent_(agency::detail::make_agent<inner_execution_agent_type>(inner_index(index), param.inner(), shared_params...))
    {}

    // friend execution_agent_traits so it has access to the constructors
    template<class> friend struct agency::execution_agent_traits;

    __AGENCY_ANNOTATION
    static outer_index_type outer_index(const index_type& index)
    {
      return __tu::tuple_head(index);
    }

    __AGENCY_ANNOTATION
    static inner_index_type inner_index(const index_type& index)
    {
      return detail::unwrap_tuple_if_not_scoped<inner_execution_category>(detail::forward_tail(index));
    }

    outer_execution_agent_type outer_agent_;
    inner_execution_agent_type inner_agent_;
};


} // end detail


template<class InnerExecutionAgent>
using sequenced_group = detail::execution_group<sequenced_agent, InnerExecutionAgent>;
template<class InnerExecutionAgent>
using sequenced_group_1d = detail::execution_group<sequenced_agent_1d, InnerExecutionAgent>;
template<class InnerExecutionAgent>
using sequenced_group_2d = detail::execution_group<sequenced_agent_2d, InnerExecutionAgent>;

template<class InnerExecutionAgent>
using parallel_group = detail::execution_group<parallel_agent, InnerExecutionAgent>;
template<class InnerExecutionAgent>
using parallel_group_1d = detail::execution_group<parallel_agent_1d, InnerExecutionAgent>;
template<class InnerExecutionAgent>
using parallel_group_2d = detail::execution_group<parallel_agent_2d, InnerExecutionAgent>;

template<class InnerExecutionAgent>
using concurrent_group = detail::execution_group<concurrent_agent, InnerExecutionAgent>;
template<class InnerExecutionAgent>
using concurrent_group_1d = detail::execution_group<concurrent_agent_1d, InnerExecutionAgent>;
template<class InnerExecutionAgent>
using concurrent_group_2d = detail::execution_group<concurrent_agent_2d, InnerExecutionAgent>;

template<class InnerExecutionAgent>
using unsequenced_group = detail::execution_group<unsequenced_agent, InnerExecutionAgent>;
template<class InnerExecutionAgent>
using unsequenced_group_1d = detail::execution_group<unsequenced_agent_1d, InnerExecutionAgent>;
template<class InnerExecutionAgent>
using unsequenced_group_2d = detail::execution_group<unsequenced_agent_2d, InnerExecutionAgent>;


namespace experimental
{
namespace detail
{


template<class BaseAgent, std::size_t static_group_size_, std::size_t static_grain_size_>
class basic_static_execution_agent : public BaseAgent
{
  private:
    using base_agent_type = BaseAgent;
    using base_traits = agency::execution_agent_traits<base_agent_type>;
    using base_param_type = typename base_traits::param_type;

  public:
    using base_agent_type::base_agent_type;

    static constexpr std::size_t static_group_size = static_group_size_;
    static constexpr std::size_t static_grain_size = static_grain_size_;

    __AGENCY_ANNOTATION
    constexpr std::size_t group_size() const
    {
      return static_group_size;
    }

    __AGENCY_ANNOTATION
    constexpr std::size_t grain_size() const
    {
      return static_grain_size;
    }

    class param_type : public base_param_type
    {
      public:
        using base_param_type::base_param_type;

        __AGENCY_ANNOTATION
        param_type() : base_param_type(0, static_group_size_) {}
    };
};


} // end detail


template<std::size_t group_size, std::size_t grain_size = 1>
using static_sequenced_agent = detail::basic_static_execution_agent<agency::sequenced_agent, group_size, grain_size>;

template<std::size_t group_size, std::size_t grain_size = 1>
using static_parallel_agent = detail::basic_static_execution_agent<agency::parallel_agent, group_size, grain_size>;

__AGENCY_ANNOTATION
constexpr std::size_t default_heap_size(std::size_t group_size)
{
  return group_size * sizeof(int);
}

template<std::size_t group_size, std::size_t grain_size = 1, std::size_t heap_size = default_heap_size(group_size)>
using static_concurrent_agent = detail::basic_static_execution_agent<
  agency::detail::basic_concurrent_agent<
    std::size_t,
    agency::detail::arena_resource<heap_size>
  >,
  group_size,
  grain_size
>;


} // end experimental
} // end agency

