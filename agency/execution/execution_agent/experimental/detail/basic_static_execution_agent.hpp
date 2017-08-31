#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/execution_agent/execution_agent_traits.hpp>
#include <cstddef>


namespace agency
{
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
} // end experimental
} // end agency

