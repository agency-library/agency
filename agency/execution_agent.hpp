#pragma once

#include <agency/execution_categories.hpp>
#include <agency/coordinate.hpp>
#include <type_traits>
#include <agency/barrier.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/unwrap_tuple_if_not_nested.hpp>
#include <agency/detail/make_tuple_if_not_nested.hpp>
#include <agency/detail/index_tuple.hpp>

namespace agency
{
namespace detail
{


__DEFINE_HAS_NESTED_TYPE(has_param_type, param_type);
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


} // end detail


template<class ExecutionAgent>
struct execution_agent_traits : detail::execution_agent_traits_base<ExecutionAgent>
{
  using execution_agent_type = ExecutionAgent;
  using execution_category = typename execution_agent_type::execution_category;
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
  //     default should be regular_grid<index_type>, but by what process should we eventually
  //     arrive at that default?
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
    template<class ExecutionAgent1>
    struct test_for_make_shared_initializer
    {
      template<
        class ExecutionAgent2,
        typename = decltype(
          ExecutionAgent2::make_shared_initializer(
            std::declval<param_type>()
          )
        )
      >
      static std::true_type test(int);

      template<class>
      static std::false_type test(...);

      using type = decltype(test<ExecutionAgent1>(0));
    };

    // XXX this should only be enabled for flat execution_agents
    //     nested agents should return a tuple of shared initializers
    template<class ExecutionAgent1>
    static decltype(std::ignore) make_shared_initializer(const param_type&, std::false_type)
    {
      return std::ignore;
    }

    template<class ExecutionAgent1>
    static auto make_shared_initializer(const param_type& param, std::true_type)
      -> decltype(
           ExecutionAgent1::make_shared_initializer(param)
         )
    {
      return ExecutionAgent1::make_shared_initializer(param);
    }

  public:

  using has_make_shared_initializer = typename test_for_make_shared_initializer<execution_agent_type>::type;

  static auto make_shared_initializer(const param_type& param)
    -> decltype(
         make_shared_initializer<execution_agent_type>(param, has_make_shared_initializer())
       )
  {
    return make_shared_initializer<execution_agent_type>(param, has_make_shared_initializer());
  }

  using shared_initializer_type = decltype(
    make_shared_initializer(std::declval<param_type>())
  );


  private:
    template<class Function, class Tuple>
    __AGENCY_ANNOTATION
    static void execute_with_shared_params_impl(Function f, const index_type& index, const param_type& param, Tuple&, std::false_type)
    {
      ExecutionAgent agent(f, index, param);
    }

    template<class Function, class Tuple>
    __AGENCY_ANNOTATION
    static void execute_with_shared_params_impl(Function f, const index_type& index, const param_type& param, Tuple& shared_params, std::true_type)
    {
      ExecutionAgent agent(f, index, param, shared_params);
    }


  public:

  template<class Function, class Tuple>
  __AGENCY_ANNOTATION
  static void execute(Function f, const index_type& index, const param_type& param, Tuple& shared_params)
  {
    execute_with_shared_params_impl(f, index, param, shared_params, has_make_shared_initializer());
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


using parallel_agent = detail::basic_execution_agent<parallel_execution_tag>;


using vector_agent = detail::basic_execution_agent<vector_execution_tag>;


class concurrent_agent : public detail::basic_execution_agent<concurrent_execution_tag>
{
  private:
    using super_t = detail::basic_execution_agent<concurrent_execution_tag>;

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
    static shared_param_type make_shared_initializer(const param_type& param)
    {
      return shared_param_type(param);
    }

  private:
    agency::barrier &barrier_;

  protected:
    template<class Function>
    concurrent_agent(Function f, const typename super_t::index_type& index, const param_type& param, shared_param_type& shared_param)
      : super_t([](super_t&){}, index, param),
        barrier_(shared_param.barrier_)
    {
      f(*this);
    }

    // friend execution_agent_traits to give it access to the constructor
    friend struct agency::execution_agent_traits<concurrent_agent>;
};


namespace detail
{


// derive from ExecutionAgent to give access to its constructor
template<class ExecutionAgent>
struct agent_access_helper : public ExecutionAgent
{
  template<class... Args>
  agent_access_helper(Args&&... args)
    : ExecutionAgent(std::forward<Args>(args)...)
  {}
};


// __make_agent helper function passes a noop functor to the agent's constructor and filters out shared parameters when necessary
template<class ExecutionAgent>
ExecutionAgent make_agent(const typename execution_agent_traits<ExecutionAgent>::index_type& index,
                          const typename execution_agent_traits<ExecutionAgent>::param_type& param)
{
  auto noop = [](ExecutionAgent&){};
  return agent_access_helper<ExecutionAgent>(noop, index, param);
}


// if an agent does not have a shared parameter, we ignore the last parameter
template<class ExecutionAgent, class T>
static ExecutionAgent make_agent(const typename execution_agent_traits<ExecutionAgent>::index_type& index,
                                 const typename execution_agent_traits<ExecutionAgent>::param_type& param,
                                 T&&,
                                 typename std::enable_if<
                                   !execution_agent_traits<ExecutionAgent>::has_make_shared_initializer::value
                                 >::type* = 0)
{
  return make_agent<ExecutionAgent>(index, param);
}


// tupled shared parameters are recieved by const reference
template<class ExecutionAgent, class Tuple>
static ExecutionAgent make_agent(const typename execution_agent_traits<ExecutionAgent>::index_type& index,
                                 const typename execution_agent_traits<ExecutionAgent>::param_type& param,
                                 const Tuple& shared_param_tuple,
                                 typename std::enable_if<
                                   execution_agent_traits<ExecutionAgent>::has_make_shared_initializer::value
                                 >::type* = 0)
{
  auto noop = [](ExecutionAgent&){};
  return agent_access_helper<ExecutionAgent>(noop, index, param, shared_param_tuple);
}


// scalar shared parameters are received by mutable reference
template<class ExecutionAgent, class T>
static ExecutionAgent make_agent(const typename execution_agent_traits<ExecutionAgent>::index_type& index,
                                 const typename execution_agent_traits<ExecutionAgent>::param_type& param,
                                 T& shared_param,
                                 typename std::enable_if<
                                   execution_agent_traits<ExecutionAgent>::has_make_shared_initializer::value
                                 >::type* = 0)
{
  auto noop = [](ExecutionAgent&){};
  return agent_access_helper<ExecutionAgent>(noop, index, param, shared_param);
}


template<class OuterExecutionAgent, class InnerExecutionAgent>
class execution_group
{
  private:
    using outer_traits = execution_agent_traits<OuterExecutionAgent>;
    using inner_traits = execution_agent_traits<InnerExecutionAgent>;

    using outer_execution_category = typename outer_traits::execution_category;
    using inner_execution_category = typename inner_traits::execution_category;

    using outer_index_type = typename execution_agent_traits<OuterExecutionAgent>::index_type;
    using inner_index_type = typename execution_agent_traits<InnerExecutionAgent>::index_type;

    // concatenates an outer index with an inner index
    // returns an index_tuple with arithmetic ops (not a std::tuple)
    static auto index_cat(const outer_index_type& outer_idx, const inner_index_type& inner_idx)
      -> decltype(
           __tu::tuple_cat_apply(
             detail::index_tuple_maker{},
             detail::make_tuple_if_not_nested<outer_execution_category>(outer_idx),
             detail::make_tuple_if_not_nested<inner_execution_category>(inner_idx)
           )
         )
    {
      return __tu::tuple_cat_apply(
        detail::index_tuple_maker{},
        detail::make_tuple_if_not_nested<outer_execution_category>(outer_idx),
        detail::make_tuple_if_not_nested<inner_execution_category>(inner_idx)
      );
    }

  public:
    using execution_category = nested_execution_tag<
      outer_execution_category,
      inner_execution_category
    >;

    using outer_execution_agent_type = OuterExecutionAgent;
    using inner_execution_agent_type = InnerExecutionAgent;

    using param_type = std::tuple<
      typename outer_traits::param_type,
      typename inner_traits::param_type
    >;

    // XXX move this into execution_agent_traits
    static auto make_shared_initializer(const param_type& param)
      -> decltype(
           std::tuple_cat(
             detail::make_tuple_if_not_nested<outer_execution_category>(
               outer_traits::make_shared_initializer(std::get<0>(param))
             ),
             detail::make_tuple_if_not_nested<inner_execution_category>(
               inner_traits::make_shared_initializer(std::get<1>(param))
             )
           )
         )
    {
      auto outer_shared_init = outer_traits::make_shared_initializer(std::get<0>(param));
      auto inner_shared_init = inner_traits::make_shared_initializer(std::get<1>(param));

      auto outer_tuple = detail::make_tuple_if_not_nested<outer_execution_category>(outer_shared_init);
      auto inner_tuple = detail::make_tuple_if_not_nested<inner_execution_category>(inner_shared_init);

      return std::tuple_cat(outer_tuple, inner_tuple);
    }

    outer_execution_agent_type& outer()
    {
      return outer_agent_;
    }

    const outer_execution_agent_type& outer() const
    {
      return outer_agent_;
    }

    inner_execution_agent_type& inner()
    {
      return inner_agent_;
    }

    const inner_execution_agent_type& inner() const
    {
      return inner_agent_;
    }

    auto index() const
      -> decltype(
           index_cat(
             std::declval<execution_group>().outer().index(),
             std::declval<execution_group>().inner().index()
           )
         )
    {
      return index_cat(this->outer().index(), this->inner().index());
    }

    using index_type = typename std::result_of<
      decltype(&execution_group::index)(execution_group)
    >::type;

    using domain_type = regular_grid<index_type>;

    domain_type domain() const
    {
      auto outer_domain = outer().domain();
      auto inner_domain = this->inner().domain();

      auto min = index_cat(outer_domain.min(), inner_domain.min());
      auto max = index_cat(outer_domain.max(), inner_domain.max());

      return domain_type{min,max};
    }

    static domain_type domain(const param_type& param)
    {
      auto outer_domain = outer_traits::domain(std::get<0>(param));
      auto inner_domain = inner_traits::domain(std::get<1>(param));

      auto min = index_cat(outer_domain.min(), inner_domain.min());
      auto max = index_cat(outer_domain.max(), inner_domain.max());

      return domain_type{min,max};
    }

    size_t group_size() const
    {
      return outer().group_size();
    }

  protected:
    template<class Function>
    execution_group(Function f, const index_type& index, const param_type& param)
      : outer_agent_(detail::make_agent<outer_execution_agent_type>(outer_index(index), std::get<0>(param))),
        inner_agent_(detail::make_agent<inner_execution_agent_type>(inner_index(index), std::get<1>(param)))
    {
      f(*this);
    }

    template<class Function, class Tuple>
    execution_group(Function f, const index_type& index, const param_type& param, Tuple& shared_param)
      : outer_agent_(detail::make_agent<outer_execution_agent_type>(outer_index(index), std::get<0>(param), __tu::tuple_head(shared_param))),
        inner_agent_(detail::make_agent<inner_execution_agent_type>(inner_index(index), std::get<1>(param), __tu::forward_tuple_tail<Tuple>(shared_param)))
    {
      f(*this);
    }

    // friend execution_agent_traits so it has access to the constructors
    template<class> friend struct agency::execution_agent_traits;

    static outer_index_type outer_index(const index_type& index)
    {
      return __tu::tuple_head(index);
    }

    static inner_index_type inner_index(const index_type& index)
    {
      return detail::unwrap_tuple_if_not_nested<inner_execution_category>(__tu::forward_tuple_tail<const index_type>(index));
    }

    outer_execution_agent_type outer_agent_;
    inner_execution_agent_type inner_agent_;
};


} // end detail


template<class InnerExecutionAgent>
using sequential_group = detail::execution_group<sequential_agent, InnerExecutionAgent>;

template<class InnerExecutionAgent>
using parallel_group = detail::execution_group<parallel_agent, InnerExecutionAgent>;

template<class InnerExecutionAgent>
using concurrent_group = detail::execution_group<concurrent_agent, InnerExecutionAgent>;

template<class InnerExecutionAgent>
using vector_group = detail::execution_group<vector_agent, InnerExecutionAgent>;


} // end agency

