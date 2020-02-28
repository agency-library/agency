#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/execution_agent/execution_agent_traits.hpp>
#include <agency/execution/executor/properties/bulk_guarantee.hpp>
#include <agency/execution/executor/properties/detail/bulk_guarantee_depth.hpp>
#include <agency/detail/tuple/tuple_utility.hpp>
#include <agency/detail/unwrap_tuple_if.hpp>
#include <agency/detail/index_tuple.hpp>
#include <agency/coordinate/detail/colexicographic_rank.hpp>

#include <utility>
#include <type_traits>


namespace agency
{
namespace detail
{


template<class OuterExecutionAgent, class Enable = void>
struct execution_group_base {};


// if execution_group's OuterExecutionAgent has a shared_param_type,
// then execution_group needs to have a shared_param_type which can be constructed from execution_group::param_type
template<class OuterExecutionAgent>
struct execution_group_base<OuterExecutionAgent,
                            typename std::enable_if<
                              detail::has_shared_param_type<OuterExecutionAgent>::value
                            >::type>
{
  struct shared_param_type : public OuterExecutionAgent::shared_param_type
  {
    template<class ParamType, class... Args>
    __AGENCY_ANNOTATION
    shared_param_type(const ParamType& param, Args&&... args)
      : OuterExecutionAgent::shared_param_type(param.outer(), std::forward<Args>(args)...)
    {}
  };
};


template<class OuterExecutionAgent, class InnerExecutionAgent>
class execution_group : public execution_group_base<OuterExecutionAgent>
{
  private:
    using outer_traits = execution_agent_traits<OuterExecutionAgent>;
    using inner_traits = execution_agent_traits<InnerExecutionAgent>;

    using outer_execution_requirement = typename outer_traits::execution_requirement;
    using inner_execution_requirement = typename inner_traits::execution_requirement;

    constexpr static size_t outer_depth = detail::bulk_guarantee_depth<outer_execution_requirement>::value;
    constexpr static size_t inner_depth = detail::bulk_guarantee_depth<inner_execution_requirement>::value;

    using outer_index_type = typename outer_traits::index_type;
    using inner_index_type = typename inner_traits::index_type;

  public:
    using index_type = detail::scoped_index_t<outer_depth, inner_depth, outer_index_type, inner_index_type>;

  private:
    // concatenates an outer index with an inner index
    // returns an index_tuple with arithmetic ops (not a std::tuple)
    __AGENCY_ANNOTATION
    static index_type make_index(const outer_index_type& outer_idx, const inner_index_type& inner_idx)
    {
      return detail::make_scoped_index<outer_depth,inner_depth>(outer_idx, inner_idx);
    }

  public:
    using execution_requirement = bulk_guarantee_t::scoped_t<
      outer_execution_requirement,
      inner_execution_requirement
    >;

    using outer_execution_agent_type = OuterExecutionAgent;
    using inner_execution_agent_type = InnerExecutionAgent;

    class param_type
    {
      private:
        typename outer_traits::param_type outer_;
        typename inner_traits::param_type inner_;

      public:
        param_type() = default;

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
      return make_index(this->outer().index(), this->inner().index());
    }

    using domain_type = lattice<index_type>;

    __AGENCY_ANNOTATION
    domain_type domain() const
    {
      auto outer_domain = outer().domain();
      auto inner_domain = this->inner().domain();

      auto min = make_index(outer_domain.min(), inner_domain.min());
      auto max = make_index(outer_domain.max(), inner_domain.max());

      return domain_type{min,max};
    }

    // XXX can probably move this to execution_agent_traits
    __AGENCY_ANNOTATION
    static domain_type domain(const param_type& param)
    {
      auto outer_domain = outer_traits::domain(param.outer());
      auto inner_domain = inner_traits::domain(param.inner());

      auto origin = make_index(outer_domain.origin(), inner_domain.origin());
      auto shape  = make_index(outer_domain.shape(), inner_domain.shape());

      return domain_type{origin,shape};
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
      return agency::detail::colexicographic_rank(index(), group_shape());
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
      // this function returns the outer_index_type contained within index
      // by simply returning its first element, or "head"
      return __tu::tuple_head(index);
    }

    __AGENCY_ANNOTATION
    static inner_index_type inner_index(const index_type& index)
    {
      // this function returns the inner_index_type contained within index
      // by taking the tail of index, and "unwrapping" it if the inner agent is not scoped
      // the inner agent is not scoped if the inner_depth is equal to one
      return detail::unwrap_tuple_if<inner_depth == 1>(detail::forward_tail(index));
    }

    outer_execution_agent_type outer_agent_;
    inner_execution_agent_type inner_agent_;
};


} // end detail
} // end agency

