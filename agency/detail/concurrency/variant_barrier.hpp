#pragma once

#include <agency/detail/config.hpp>
#include <agency/experimental/variant.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/type_list.hpp>


namespace agency
{
namespace detail
{


template<class Barrier, class... Barriers>
class variant_barrier : private agency::experimental::variant<Barrier, Barriers..., agency::experimental::monostate>
{
  private:
    using variant_type = agency::experimental::variant<Barrier, Barriers..., agency::experimental::monostate>;
    using super_t = variant_type;

  public:
    template<class T,
             __AGENCY_REQUIRES(agency::detail::disjunction<std::is_same<T,Barrier>, std::is_same<T,Barriers>...>::value)
            >
    __AGENCY_ANNOTATION
    variant_barrier(agency::experimental::in_place_type_t<T> which, size_t count)
      : super_t(which, count)
    {}

  private:
    __AGENCY_ANNOTATION
    void dynamic_emplace(agency::detail::type_list<>, size_t, size_t)
    {
      assert(0);
    }

    template<class T, class... Types>
    __AGENCY_ANNOTATION
    void dynamic_emplace(agency::detail::type_list<T,Types...>, size_t index, size_t count)
    {
      if(index == 0)
      {
        super_t::template emplace<T>(count);
      }
      else
      {
        dynamic_emplace(agency::detail::type_list<Types...>(), index-1,count);
      }
    }

  public:
    __AGENCY_ANNOTATION
    variant_barrier(size_t index, size_t count)
      : super_t(agency::experimental::monostate{})
    {
      dynamic_emplace(agency::detail::type_list<Barrier,Barriers...>(), index, count);
    }

    using super_t::index;

  private:
    struct count_visitor
    {
      __agency_exec_check_disable__
      template<class T>
      __AGENCY_ANNOTATION
      size_t operator()(const T& self) const
      {
        return self.count();
      }

      __agency_exec_check_disable__
      __AGENCY_ANNOTATION
      size_t operator()(agency::experimental::monostate) const
      {
        assert(0);
        return 0;
      }
    };

  public:
    __AGENCY_ANNOTATION
    size_t count() const
    {
      return agency::experimental::visit(count_visitor{}, static_cast<const super_t&>(*this));
    }

  private:
    struct arrive_and_wait_visitor
    {
      __agency_exec_check_disable__
      template<class T>
      __AGENCY_ANNOTATION
      void operator()(T& self) const
      {
        self.arrive_and_wait();
      }

      __AGENCY_ANNOTATION
      void operator()(agency::experimental::monostate) const
      {
        assert(0);
      }
    };

  public:
    __AGENCY_ANNOTATION
    void arrive_and_wait()
    {
      agency::experimental::visit(arrive_and_wait_visitor{}, static_cast<super_t&>(*this));
    }
};


} // end detail
} // end agency

