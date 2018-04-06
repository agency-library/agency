// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/executor/properties/detail/static_query.hpp>
#include <agency/execution/executor/executor_traits/detail/can_query.hpp>


namespace agency
{


// declare bulk_guarantee_t for detail::executor_with_bulk_guarantee
struct bulk_guarantee_t;
  

namespace detail
{


template<class Executor, class BulkGuarantee>
class executor_with_bulk_guarantee : public Executor
{
  private:
    using super_t = Executor;

  public:
    __AGENCY_ANNOTATION
    constexpr explicit executor_with_bulk_guarantee(const Executor& ex)
      : super_t{ex}
    {}

    __AGENCY_ANNOTATION
    constexpr static BulkGuarantee query(const bulk_guarantee_t&)
    {
      return BulkGuarantee();
    }

    // forward all other queries to the base class

    // static query member function
    __agency_exec_check_disable__
    template<class Property,
             class E = Executor,
             __AGENCY_REQUIRES(
               has_static_query<Property, E>::value
             )>
    __AGENCY_ANNOTATION
    constexpr static auto query(const Property& p) ->
      decltype(Property::template static_query<E>())
    {
      return Property::template static_query<E>();
    }

    // non-static query member function
    __agency_exec_check_disable__
    template<class Property,
             class E = Executor,
             __AGENCY_REQUIRES(
               !has_static_query<Property, E>::value and
               has_query_member<E,Property>::value
             )>
    __AGENCY_ANNOTATION
    constexpr auto query(const Property& p) const ->
      decltype(std::declval<const E>().query(p))
    {
      return super_t::query(p);
    }
};


} // end detail


struct bulk_guarantee_t
{
  static constexpr bool is_requirable = false;
  static constexpr bool is_preferable = false;

  template<class E>
  __AGENCY_ANNOTATION
  static constexpr auto static_query() ->
    decltype(detail::static_query<E,bulk_guarantee_t>())
  {
    return detail::static_query<E,bulk_guarantee_t>();
  }

  __AGENCY_ANNOTATION
  friend constexpr bool operator==(const bulk_guarantee_t& a, const bulk_guarantee_t& b)
  {
    return a.which_ == b.which_;
  }

  __AGENCY_ANNOTATION
  friend constexpr bool operator!=(const bulk_guarantee_t& a, const bulk_guarantee_t& b)
  {
    return !(a == b);
  }

  __AGENCY_ANNOTATION
  constexpr bulk_guarantee_t()
    : which_{0}
  {}

  struct sequenced_t
  {
    static constexpr bool is_requirable = true;
    static constexpr bool is_preferable = true;

    template<class Executor>
    static constexpr auto static_query() ->
      decltype(detail::static_query<Executor,sequenced_t>())
    {
      return detail::static_query<Executor,sequenced_t>();
    }
    
    __AGENCY_ANNOTATION
    static constexpr sequenced_t value()
    {
      return sequenced_t{};
    }
  };

  static constexpr sequenced_t sequenced{};

  __AGENCY_ANNOTATION
  constexpr bulk_guarantee_t(const sequenced_t&)
    : which_{1}
  {}


  struct concurrent_t
  {
    static constexpr bool is_requirable = true;
    static constexpr bool is_preferable = true;

    template<class Executor>
    static constexpr auto static_query() ->
      decltype(detail::static_query<Executor,concurrent_t>())
    {
      return detail::static_query<Executor,concurrent_t>();
    }

    __AGENCY_ANNOTATION
    static constexpr concurrent_t value()
    {
      return concurrent_t{};
    }
  };

  static constexpr concurrent_t concurrent{};

  __AGENCY_ANNOTATION
  constexpr bulk_guarantee_t(const concurrent_t&)
    : which_{2}
  {}


  struct parallel_t
  {
    static constexpr bool is_requirable = true;
    static constexpr bool is_preferable = true;

    template<class Executor>
    static constexpr auto static_query() ->
      decltype(detail::static_query<Executor,parallel_t>())
    {
      return detail::static_query<Executor,parallel_t>();
    }
    
    __AGENCY_ANNOTATION
    static constexpr parallel_t value()
    {
      return parallel_t{};
    }

    template<class Executor,
             __AGENCY_REQUIRES(
               detail::static_query<Executor, sequenced_t>() == sequenced_t() or // sequenced executors should be adapted
               detail::static_query<Executor, concurrent_t>() == concurrent_t()  // concurrent executors should be adapted
             )>
    __AGENCY_ANNOTATION
    friend constexpr detail::executor_with_bulk_guarantee<Executor,parallel_t> require(const Executor& ex, parallel_t)
    {
      return detail::executor_with_bulk_guarantee<Executor,parallel_t>{ex};
    }
  };

  static constexpr parallel_t parallel{};

  __AGENCY_ANNOTATION
  constexpr bulk_guarantee_t(const parallel_t&)
    : which_{3}
  {}


  struct unsequenced_t
  {
    static constexpr bool is_requirable = true;
    static constexpr bool is_preferable = true;

    template<class Executor>
    static constexpr auto static_query() ->
      decltype(detail::static_query<Executor,unsequenced_t>())
    {
      return detail::static_query<Executor,unsequenced_t>();
    }

    template<class Executor,
             __AGENCY_REQUIRES(
               !detail::can_query<Executor, sequenced_t>::value and
               !detail::can_query<Executor, parallel_t>::value and
               !detail::has_query_member<Executor, unsequenced_t>::value
             )>
    static constexpr unsequenced_t static_query()
    {
      return unsequenced_t{};
    }
    
    __AGENCY_ANNOTATION
    static constexpr unsequenced_t value()
    {
      return unsequenced_t{};
    }

    template<class Executor>
    __AGENCY_ANNOTATION
    friend constexpr detail::executor_with_bulk_guarantee<Executor,unsequenced_t> require(const Executor& ex, unsequenced_t)
    {
      return detail::executor_with_bulk_guarantee<Executor,unsequenced_t>{ex};
    }
  };

  static constexpr unsequenced_t unsequenced{};

  __AGENCY_ANNOTATION
  constexpr bulk_guarantee_t(const unsequenced_t&)
    : which_{4}
  {}


  // By default, executors are unsequenced if a bulk_guarantee_t cannot be queried through a member
  template<class Executor,
           __AGENCY_REQUIRES(
             !detail::has_query_member<Executor, bulk_guarantee_t>::value
           )>
  static constexpr unsequenced_t static_query()
  {
    return unsequenced_t{};
  }

  template<class OuterGuarantee, class InnerGuarantee>
  class scoped_t
  {
    public:
      static constexpr bool is_requirable = true;
      static constexpr bool is_preferable = true;
  
      template<class Executor>
      __AGENCY_ANNOTATION
      static constexpr auto static_query() ->
        decltype(detail::static_query<Executor, scoped_t>())
      {
        return detail::static_query<Executor, scoped_t>();
      }

      scoped_t() = default;

      scoped_t(const scoped_t&) = default;
  
      __AGENCY_ANNOTATION
      constexpr scoped_t(const OuterGuarantee& outer, const InnerGuarantee& inner)
        : outer_{outer}, inner_{inner}
      {}
  
      __AGENCY_ANNOTATION
      constexpr OuterGuarantee outer() const
      {
        return outer_;
      }
  
      __AGENCY_ANNOTATION
      constexpr InnerGuarantee inner() const
      {
        return inner_;
      }
  
      __AGENCY_ANNOTATION
      constexpr scoped_t value() const
      {
        return *this;
      }
  
      __AGENCY_ANNOTATION
      friend constexpr bool operator==(const scoped_t& a, const scoped_t& b)
      {
        return (a.outer().value() == b.outer().value()) && (a.inner().value() == b.inner().value());
      }
  
      __AGENCY_ANNOTATION
      friend constexpr bool operator!=(const scoped_t& a, const scoped_t& b)
      {
        return !(a == b);
      }

    private:
      OuterGuarantee outer_;
      InnerGuarantee inner_;
  }; // end scoped_t

  template<class OuterGuarantee, class InnerGuarantee>
  __AGENCY_ANNOTATION
  constexpr static scoped_t<OuterGuarantee, InnerGuarantee>
    scoped(const OuterGuarantee& outer, const InnerGuarantee& inner)
  {
    return {outer, inner};
  }

  private:
    unsigned int which_;
}; // end bulk_guarantee_t


namespace
{


// define the property object

#ifndef __CUDA_ARCH__
constexpr bulk_guarantee_t bulk_guarantee{};
#else
// CUDA __device__ functions cannot access global variables so make bulk_guarantee a __device__ variable in __device__ code
const __device__ bulk_guarantee_t bulk_guarantee;
#endif


} // end anonymous namespace


namespace detail
{


// XXX maybe "weakness" is the wrong way to describe what we're really interested in here
//     we just want some idea of substitutability

//  "<" means "is weaker than"
//  "<" is transitive
// if guarantee A is weaker than guarantee B,
// then agents with guarantee A can be executed with agents with guarantee B
//
// these relationships should be true
//
// parallel_t    < sequenced_t
// parallel_t    < concurrent_t
// unsequenced_t < parallel_t
//
// XXX figure out how scoped_t sorts

// in general, one guarantee is not weaker than another
template<class BulkGuarantee1, class BulkGuarantee2>
struct is_weaker_guarantee_than : std::false_type {};

// all guarantees are weaker than themselves
template<class BulkGuarantee>
struct is_weaker_guarantee_than<BulkGuarantee, BulkGuarantee> : std::true_type {};

// unsequenced is weaker than everything else
template<class BulkGuarantee>
struct is_weaker_guarantee_than<bulk_guarantee_t::unsequenced_t, BulkGuarantee> : std::true_type {};

// introduce this specialization to disambiguate other specializations
template<>
struct is_weaker_guarantee_than<bulk_guarantee_t::unsequenced_t, bulk_guarantee_t::unsequenced_t> : std::true_type {};

// parallel is weaker than sequenced & concurrent
template<>
struct is_weaker_guarantee_than<bulk_guarantee_t::parallel_t, bulk_guarantee_t::sequenced_t> : std::true_type {};

template<>
struct is_weaker_guarantee_than<bulk_guarantee_t::parallel_t, bulk_guarantee_t::concurrent_t> : std::true_type {};

} // end detail


} // end agency

