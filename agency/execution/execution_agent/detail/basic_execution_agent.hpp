#pragma once

#include <agency/detail/config.hpp>
#include <agency/coordinate/lattice.hpp>
#include <agency/detail/index_lexicographical_rank.hpp>
#include <agency/execution/execution_agent/execution_agent_traits.hpp>
#include <utility>


namespace agency
{
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
      return agency::detail::index_lexicographical_rank(index(), group_shape());
    }

    __AGENCY_ANNOTATION
    bool elect() const
    {
      return rank() == 0;
    }

    class param_type
    {
      public:
        param_type() = default;

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
} // end agency

