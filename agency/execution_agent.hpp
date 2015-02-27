#pragma once

#include <agency/execution_categories.hpp>
#include <agency/barrier.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/index_tuple.hpp>
#include <agency/detail/unwrap_tuple_if_not_nested.hpp>
#include <agency/detail/make_tuple_if_not_nested.hpp>
#include <agency/coordinate.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{


__DEFINE_HAS_NESTED_TYPE(has_param_type, param_type);
__DEFINE_HAS_NESTED_TYPE(has_shared_param_type, shared_param_type);
__DEFINE_HAS_NESTED_TYPE(has_inner_execution_agent_type, inner_execution_agent_type);


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
  //     default should be regular_grid<index_type>, but by what process should we eventually
  //     arrive at that default?
  // XXX yank the general implementation from execution_group now that param_type::inner() exists
  __agency_hd_warning_disable__
  __AGENCY_ANNOTATION
  static auto domain(const param_type& param)
    -> decltype(ExecutionAgent::domain(param))
  {
    return ExecutionAgent::domain(param);
  }

  using domain_type = decltype(domain(std::declval<param_type>()));

  template<class Function>
  __AGENCY_ANNOTATION
  static void execute(Function f, const index_type& index, const param_type& param)
  {
    ExecutionAgent agent(f, index, param);
  }


  private:
    template<class T>
    struct execution_agent_shared_param
    {
      using type = typename T::shared_param_type;
    };

    template<class Function, class... Args>
    __AGENCY_ANNOTATION
    static void execute_with_shared_params_impl(std::false_type, Function f, const index_type& index, const param_type& param, Args&&...)
    {
      ExecutionAgent agent(f, index, param);
    }

    template<class Function, class... Args>
    __AGENCY_ANNOTATION
    static void execute_with_shared_params_impl(std::true_type, Function f, const index_type& index, const param_type& param, Args&... shared_params)
    {
      ExecutionAgent agent(f, index, param, shared_params...);
    }

  public:

  using shared_param_type = typename detail::lazy_conditional<
    detail::has_shared_param_type<execution_agent_type>::value,
    execution_agent_shared_param<execution_agent_type>,
    detail::identity<agency::detail::ignore_t>
  >::type;


  // XXX should ensure that the SharedParams are all shared_param_type &
  template<class Function, class... SharedParams>
  __AGENCY_ANNOTATION
  static void execute(Function f, const index_type& index, const param_type& param, shared_param_type& shared_param1, SharedParams&... shared_params)
  {
    execute_with_shared_params_impl(typename detail::has_shared_param_type<execution_agent_type>::type(), f, index, param, shared_param1, shared_params...);
  }


  private:
    template<class Function, class Tuple, size_t... Indices>
    __AGENCY_ANNOTATION
    static void unpack_shared_params_and_execute(Function f, const index_type& index, const param_type& param, Tuple& shared_params, detail::index_sequence<Indices...>)
    {
      execute(f, index, param, detail::get<Indices>(shared_params)...);
    }


  public:

  // XXX should ensure that the shared_params are all the right type and are references
  template<class Function, class Tuple>
  __AGENCY_ANNOTATION
  static void execute(Function f, const index_type& index, const param_type& param, Tuple& shared_params)
  {
    unpack_shared_params_and_execute(f, index, param, shared_params, detail::make_index_sequence<std::tuple_size<Tuple>::value>());
  }


  private:
    template<class ExecutionAgent1>
    struct test_for_make_shared_param_tuple
    {
      template<
        class ExecutionAgent2,
        typename = decltype(
          ExecutionAgent2::make_shared_param_tuple(
            std::declval<param_type>()
          )
        )
      >
      static std::true_type test(int);

      template<class>
      static std::false_type test(...);

      using type = decltype(test<ExecutionAgent1>(0));
    };

    using has_make_shared_param_tuple = typename test_for_make_shared_param_tuple<execution_agent_type>::type;


    template<class TypeList>
    struct default_execution_agent_shared_param_tuple_impl;


    template<class... ExecutionAgents>
    struct default_execution_agent_shared_param_tuple_impl<detail::type_list<ExecutionAgents...>>
    {
      using type = detail::tuple<
        typename execution_agent_traits<ExecutionAgents>::shared_param_type...
      >;
    };


    template<class ExecutionAgent1>
    struct default_execution_agent_shared_param_tuple : default_execution_agent_shared_param_tuple_impl<
      typename detail::execution_agent_type_list<ExecutionAgent1>::type
    >
    {};


    template<class ExecutionAgent1>
    struct result_of_make_shared_param_tuple
    {
      using param_type = typename execution_agent_traits<ExecutionAgent1>::param_type;
      using type = decltype(ExecutionAgent1::make_shared_param_tuple(std::declval<param_type>()));
    };


    using shared_param_tuple_type = typename detail::lazy_conditional<
      has_make_shared_param_tuple::value,
      result_of_make_shared_param_tuple<execution_agent_type>,
      default_execution_agent_shared_param_tuple<execution_agent_type>
    >::type;


    template<class ExecutionAgent1>
    __AGENCY_ANNOTATION
    static shared_param_type make_shared_param(const param_type& param,
                                               typename std::enable_if<
                                                 detail::has_shared_param_type<ExecutionAgent1>::value
                                               >::type* = 0)
    {
      return shared_param_type{param};
    }


    template<class ExecutionAgent1>
    __AGENCY_ANNOTATION
    static shared_param_type make_shared_param(const param_type& param,
                                               typename std::enable_if<
                                                 !detail::has_shared_param_type<ExecutionAgent1>::value
                                               >::type* = 0)
    {
      return agency::detail::ignore;
    }


    // default case for flat agents
    template<class ExecutionAgent1>
    __AGENCY_ANNOTATION
    static shared_param_tuple_type make_shared_param_tuple_default_impl(const param_type& param, std::false_type)
    {
      return detail::make_tuple(make_shared_param<ExecutionAgent1>(param));
    }


    // default case for nested agents
    template<class ExecutionAgent1>
    __AGENCY_ANNOTATION
    static shared_param_tuple_type make_shared_param_tuple_default_impl(const param_type& param, std::true_type)
    {
      using inner_traits = execution_agent_traits<
        typename ExecutionAgent1::inner_execution_agent_type
      >;

      // recurse to get the tail of the tuple
      auto inner_params = inner_traits::make_shared_param_tuple(param.inner());

      // prepend the head 
      return __tu::tuple_prepend_invoke(inner_params, make_shared_param<ExecutionAgent1>(param), detail::agency_tuple_maker());
    }


    template<class ExecutionAgent1>
    __AGENCY_ANNOTATION
    static shared_param_tuple_type make_shared_param_tuple_impl(const param_type& param, std::false_type)
    {
      // the execution agent does not have the function, so use the default implementation
      return make_shared_param_tuple_default_impl<ExecutionAgent1>(param, typename detail::has_inner_execution_agent_type<ExecutionAgent1>::type());
    }


    __agency_hd_warning_disable__
    template<class ExecutionAgent1>
    __AGENCY_ANNOTATION
    static shared_param_tuple_type make_shared_param_tuple_impl(const param_type& param, std::true_type)
    {
      // the execution agent has the function, so just call it
      return ExecutionAgent1::make_shared_param_tuple(param);
    }


  public:
    __AGENCY_ANNOTATION
    static shared_param_tuple_type make_shared_param_tuple(const param_type& param)
    {
      return make_shared_param_tuple_impl<execution_agent_type>(param, has_make_shared_param_tuple());
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

    using domain_type = regular_grid<index_type>;

    __AGENCY_ANNOTATION
    const domain_type& domain() const
    {
      return domain_;
    }

    __AGENCY_ANNOTATION
    auto group_size() const
      -> decltype(this->domain().size())
    {
      return domain().size();
    }

    __AGENCY_ANNOTATION
    auto group_shape() const
      -> decltype(this->domain().shape())
    {
      return domain().shape();
    }

    class param_type
    {
      public:
        __AGENCY_ANNOTATION
        param_type() = default;

        __AGENCY_ANNOTATION
        param_type(const param_type& other)
          : domain_(other.domain_)
        {}

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
    __agency_hd_warning_disable__
    template<class Function>
    __AGENCY_ANNOTATION
    basic_execution_agent(Function f, const index_type& index, const param_type& param)
      : index_(index),
        domain_(param.domain())
    {
      f(*this);
    }

    friend struct agency::execution_agent_traits<basic_execution_agent>;

  private:
    index_type index_;
    domain_type domain_;
};


} // end detail


using sequential_agent = detail::basic_execution_agent<sequential_execution_tag>;
using sequential_agent_1d = sequential_agent;
using sequential_agent_2d = detail::basic_execution_agent<sequential_execution_tag, size2>;


using parallel_agent = detail::basic_execution_agent<parallel_execution_tag>;
using parallel_agent_1d = parallel_agent;
using parallel_agent_2d = detail::basic_execution_agent<parallel_execution_tag, size2>;


using vector_agent = detail::basic_execution_agent<vector_execution_tag>;
using vector_agent_1d = vector_agent;
using vector_agent_2d = detail::basic_execution_agent<vector_execution_tag, size2>;


namespace detail
{



template<class Index>
class basic_concurrent_agent : public detail::basic_execution_agent<concurrent_execution_tag, Index>
{
  private:
    using super_t = detail::basic_execution_agent<concurrent_execution_tag, Index>;

  public:
    void wait() const
    {
      barrier_.count_down_and_wait();
    }

    struct shared_param_type
    {
      shared_param_type(const typename super_t::param_type& param)
        : count_(param.domain().size()),
          barrier_(count_)
      {}

      shared_param_type(const shared_param_type& other)
        : count_(other.count_),
          barrier_(count_)
      {}

      int count_;
      agency::barrier barrier_;
    };

    // XXX seems like we either need shared_param_type or make_shared_initializer()
    //     but not both
    //     if execution_agent_traits checks for the existence of shared_param_type,
    //     can't it just call its constructor?
    static shared_param_type make_shared_initializer(const typename super_t::param_type& param)
    {
      return shared_param_type(param);
    }

  private:
    agency::barrier &barrier_;

  protected:
    template<class Function>
    basic_concurrent_agent(Function f, const typename super_t::index_type& index, const typename super_t::param_type& param, shared_param_type& shared_param)
      : super_t([](super_t&){}, index, param),
        barrier_(shared_param.barrier_)
    {
      f(*this);
    }

    // friend execution_agent_traits to give it access to the constructor
    friend struct agency::execution_agent_traits<basic_concurrent_agent>;
};


} // end detail


using concurrent_agent = detail::basic_concurrent_agent<size_t>;
using concurrent_agent_1d = concurrent_agent;
using concurrent_agent_2d = detail::basic_concurrent_agent<size2>;


namespace detail
{


// derive from ExecutionAgent to give access to its constructor
template<class ExecutionAgent>
struct agent_access_helper : public ExecutionAgent
{
  struct noop
  {
    __AGENCY_ANNOTATION void operator()(ExecutionAgent&) {}
  };

  template<class... Args>
  __AGENCY_ANNOTATION
  agent_access_helper(Args&&... args)
    : ExecutionAgent(noop(), std::forward<Args>(args)...)
  {}
};


// __make_agent helper function passes a noop functor to the agent's constructor and filters out shared parameters when necessary
template<class ExecutionAgent>
__AGENCY_ANNOTATION
ExecutionAgent make_agent(const typename execution_agent_traits<ExecutionAgent>::index_type& index,
                          const typename execution_agent_traits<ExecutionAgent>::param_type& param)
{
  return agent_access_helper<ExecutionAgent>(index, param);
}


template<class ExecutionAgent, class SharedParam>
__AGENCY_ANNOTATION
static ExecutionAgent make_flat_agent(const typename execution_agent_traits<ExecutionAgent>::index_type& index,
                                      const typename execution_agent_traits<ExecutionAgent>::param_type& param,
                                      SharedParam&,
                                      typename std::enable_if<
                                        std::is_same<SharedParam,agency::detail::ignore_t>::value
                                      >::type* = 0)
{
  return make_agent<ExecutionAgent>(index, param);
}


template<class ExecutionAgent, class SharedParam>
__AGENCY_ANNOTATION
static ExecutionAgent make_flat_agent(const typename execution_agent_traits<ExecutionAgent>::index_type& index,
                                      const typename execution_agent_traits<ExecutionAgent>::param_type& param,
                                      SharedParam& shared_param,
                                      typename std::enable_if<
                                        !std::is_same<SharedParam,agency::detail::ignore_t>::value
                                      >::type* = 0)
{
  return agent_access_helper<ExecutionAgent>(index, param, shared_param);
}


template<class ExecutionAgent>
__AGENCY_ANNOTATION
static ExecutionAgent make_agent(const typename execution_agent_traits<ExecutionAgent>::index_type& index,
                                 const typename execution_agent_traits<ExecutionAgent>::param_type& param,
                                 typename execution_agent_traits<ExecutionAgent>::shared_param_type& shared_param)
{
  return make_flat_agent<ExecutionAgent>(index, param, shared_param);
}


template<class ExecutionAgent, class SharedParam2, class... SharedParams>
__AGENCY_ANNOTATION
static ExecutionAgent make_agent(const typename execution_agent_traits<ExecutionAgent>::index_type& index,
                                 const typename execution_agent_traits<ExecutionAgent>::param_type& param,
                                 typename execution_agent_traits<ExecutionAgent>::shared_param_type& shared_param1,
                                 SharedParam2&    shared_param2,
                                 SharedParams&... shared_params)
{
  return agent_access_helper<ExecutionAgent>(index, param, shared_param1, shared_param2, shared_params...);
}


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
        agency::detail::make_tuple_if_not_nested<outer_execution_category>(std::declval<outer_index_type>()),
        agency::detail::make_tuple_if_not_nested<inner_execution_category>(std::declval<inner_index_type>())
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
        agency::detail::make_tuple_if_not_nested<outer_execution_category>(outer_idx),
        agency::detail::make_tuple_if_not_nested<inner_execution_category>(inner_idx)
      );
    }

  public:
    using execution_category = nested_execution_tag<
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

    using domain_type = regular_grid<index_type>;

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
    size_t group_size() const
    {
      return outer().group_size();
    }

  protected:
    __agency_hd_warning_disable__
    template<class Function>
    __AGENCY_ANNOTATION
    execution_group(Function f, const index_type& index, const param_type& param)
      : outer_agent_(detail::make_agent<outer_execution_agent_type>(outer_index(index), param.outer())),
        inner_agent_(detail::make_agent<inner_execution_agent_type>(inner_index(index), param.inner()))
    {
      f(*this);
    }

    // XXX ensure all the shared params are the right type
    __agency_hd_warning_disable__
    template<class Function, class SharedParam1, class... SharedParams>
    __AGENCY_ANNOTATION
    execution_group(Function f, const index_type& index, const param_type& param, SharedParam1& shared_param1, SharedParams&... shared_params)
      : outer_agent_(agency::detail::make_agent<outer_execution_agent_type>(outer_index(index), param.outer(), shared_param1)),
        inner_agent_(agency::detail::make_agent<inner_execution_agent_type>(inner_index(index), param.inner(), shared_params...))
    {
      f(*this);
    }

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
      return detail::unwrap_tuple_if_not_nested<inner_execution_category>(detail::forward_tail(index));
    }

    outer_execution_agent_type outer_agent_;
    inner_execution_agent_type inner_agent_;
};


} // end detail


template<class InnerExecutionAgent>
using sequential_group = detail::execution_group<sequential_agent, InnerExecutionAgent>;
template<class InnerExecutionAgent>
using sequential_group_1d = detail::execution_group<sequential_agent_1d, InnerExecutionAgent>;
template<class InnerExecutionAgent>
using sequential_group_2d = detail::execution_group<sequential_agent_2d, InnerExecutionAgent>;

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
using vector_group = detail::execution_group<vector_agent, InnerExecutionAgent>;
template<class InnerExecutionAgent>
using vector_group_1d = detail::execution_group<vector_agent_1d, InnerExecutionAgent>;
template<class InnerExecutionAgent>
using vector_group_2d = detail::execution_group<vector_agent_2d, InnerExecutionAgent>;


} // end agency

