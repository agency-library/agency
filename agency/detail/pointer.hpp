#pragma once

#include <agency/detail/tuple.hpp>

namespace agency
{
namespace detail
{


template<class... Pointers>
class zip_pointer;


} // end detail
} // end agency


namespace std
{
// specializations of stuff in std come before their use

template<size_t i, class... Types>
struct tuple_element<i, agency::detail::zip_pointer<Types...>> : std::tuple_element<i,agency::detail::tuple<Types...>> {};

template<class... Types>
struct tuple_size<agency::detail::zip_pointer<Types...>> : std::tuple_size<agency::detail::tuple<Types...>> {};


} // end std


namespace agency
{
namespace detail
{


// like a really dumb zip_iterator
// all it can do is dereference
template<class... Pointers>
class zip_pointer : public agency::detail::tuple<Pointers...>
{
  private:
    using super_t = agency::detail::tuple<Pointers...>;

  public:
    using super_t::super_t;

    __host__ __device__
    zip_pointer(const super_t& other) : super_t(other) {}

    using reference = agency::detail::tuple<
      decltype(*std::declval<Pointers>())...
    >;

  private:
    template<size_t... Indices>
    __host__ __device__
    reference dereference_impl(agency::detail::index_sequence<Indices...>) const
    {
      return reference(*agency::detail::get<Indices>(*this)...);
    }

  public:
    // implement dereference operator to return a tuple of references
    __host__ __device__
    reference operator*() const
    {
      return dereference_impl(agency::detail::index_sequence_for<Pointers...>());
    }
};


} // end detail
} // end agency

