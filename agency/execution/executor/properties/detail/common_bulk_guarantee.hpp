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
#include <agency/execution/executor/properties/bulk_guarantee.hpp>
#include <agency/execution/executor/properties/detail/bulk_guarantee_depth.hpp>


namespace agency
{
namespace detail
{
namespace common_bulk_guarantee_detail
{


// because the implementation of common_bulk_guarantee2 is recursive, we introduce
// a forward declaration so it may call itself.
template<class BulkGuarantee1, class BulkGuarantee2>
struct common_bulk_guarantee2;

template<class BulkGuarantee1, class BulkGuarantee2>
using common_bulk_guarantee2_t = typename common_bulk_guarantee2<BulkGuarantee1,BulkGuarantee2>::type;


// the implementation of common_bulk_guarantee2_impl is recursive, and there are two base cases

// base case 1: the input guarantees have different depths
// i.e., one may be "flat" and the other scoped,
// or both may be scoped but have different depths
template<class BulkGuarantee1, class BulkGuarantee2, size_t depth1, size_t depth2>
struct common_bulk_guarantee2_impl
{
  // there's no commonality between the two input guarantees
  // so the result is unsequenced -- there is no other static guarantee that can be provided
  using type = bulk_guarantee_t::unsequenced_t;
};


// base case 2: both input guarantees are "flat"
template<class BulkGuarantee1, class BulkGuarantee2>
struct common_bulk_guarantee2_impl<BulkGuarantee1,BulkGuarantee2,1,1>
{
  // both BulkGuarantee1 & BulkGuarantee2 have depth 1 -- they are "flat"

  // if one of the two guarantees is weaker than the other, then return it
  // otherwise, return the weakest static guarantee: unsequenced
  using type = conditional_t<
    is_weaker_guarantee_than<BulkGuarantee1,BulkGuarantee2>::value,
    BulkGuarantee1,
    conditional_t<
      is_weaker_guarantee_than<BulkGuarantee2,BulkGuarantee1>::value,
      BulkGuarantee2,
      bulk_guarantee_t::unsequenced_t
    >
  >;
};


// recursive case: both input guarantees are scoped
template<class OuterGuarantee1, class InnerGuarantee1, class OuterGuarantee2, class InnerGuarantee2, size_t depth>
struct common_bulk_guarantee2_impl<
  bulk_guarantee_t::scoped_t<OuterGuarantee1,InnerGuarantee1>,
  bulk_guarantee_t::scoped_t<OuterGuarantee2,InnerGuarantee2>,
  depth,depth
>
{
  // both guarantees are scoped and they have the same depth
  // XXX it may not matter so much that the depth is the same.
  //     we may still be able to apply this recipe sensibly even
  //     when the two input guarantees' depths differ

  // the result is scoped. apply common_bulk_guarantee to
  // the inputs' constituents to get the result's constituents
  using type = bulk_guarantee_t::scoped_t<
    common_bulk_guarantee2_t<OuterGuarantee1,OuterGuarantee2>,
    common_bulk_guarantee2_t<InnerGuarantee1,InnerGuarantee2>
  >;
};


template<class BulkGuarantee1, class BulkGuarantee2>
struct common_bulk_guarantee2
{
  using type = typename common_bulk_guarantee2_impl<
    BulkGuarantee1,
    BulkGuarantee2,
    bulk_guarantee_depth<BulkGuarantee1>::value,
    bulk_guarantee_depth<BulkGuarantee2>::value
  >::type;
};

} // end common_bulk_guarantee_detail


// common_bulk_guarantee is a type trait which, given one or more possibly different bulk guarantee types,
// returns a type representing the strongest guarantees that can be made given the different input possibilities
template<class BulkGuarantee, class... BulkGuarantees>
struct common_bulk_guarantee;

template<class BulkGuarantee, class... BulkGuarantees>
using common_bulk_guarantee_t = typename common_bulk_guarantee<BulkGuarantee,BulkGuarantees...>::type;


// the implementation of common_bulk_guarantee is recursive
// this is the recursive case
template<class BulkGuarantee1, class BulkGuarantee2, class... BulkGuarantees>
struct common_bulk_guarantee<BulkGuarantee1, BulkGuarantee2, BulkGuarantees...>
{
  using type = common_bulk_guarantee_t<
    BulkGuarantee1,
    common_bulk_guarantee_t<BulkGuarantee2, BulkGuarantees...>
  >;
};

// base case 1: a single guarantee
template<class BulkGuarantee>
struct common_bulk_guarantee<BulkGuarantee>
{
  using type = BulkGuarantee;
};

// base case 2: two guarantees
template<class BulkGuarantee1, class BulkGuarantee2>
struct common_bulk_guarantee<BulkGuarantee1,BulkGuarantee2>
{
  // with two guarantees, we lower onto the two guarantee
  // implementation inside common_bulk_guarantee_detail
  using type = common_bulk_guarantee_detail::common_bulk_guarantee2_t<BulkGuarantee1,BulkGuarantee2>;
};


} // end detail
} // end agency

