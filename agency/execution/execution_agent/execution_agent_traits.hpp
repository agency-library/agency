#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/has_member.hpp>
#include <agency/tuple.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/type_list.hpp>
#include <type_traits>
#include <utility>


namespace agency
{
namespace detail
{


__DEFINE_HAS_MEMBER_TYPE(has_param_type, param_type);
__DEFINE_HAS_MEMBER_TYPE(has_shared_param_type, shared_param_type);
__DEFINE_HAS_MEMBER_TYPE(has_inner_execution_agent_type, inner_execution_agent_type);


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
                                      detail::ignore_t)
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
  using execution_requirement = typename execution_agent_type::execution_requirement;

  // XXX we should probably use execution_agent_type::index_type if it exists,
  //     if not, use the type of the result of .index()
  using index_type = detail::decay_t<
    decltype(
      std::declval<execution_agent_type>().index()
    )
  >;

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
      detail::identity<detail::ignore_t>
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


} // end agency

