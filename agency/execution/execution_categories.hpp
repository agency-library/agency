/// \file
/// \brief Contains definitions of execution categories.
///

/// \defgroup execution_categories Execution Categories
/// \ingroup execution
/// \brief Execution categories categorize forward progress behavior.
///
/// Execution categories describe the forward progress behavior of groups of execution agents
/// without regard to the thread on which an individual agent executes.
///
/// Each execution agent within a group of such agents is associated with a function invocation.
/// This function invocation depends on the context in which execution agents
/// are created; for example, the group of function invocations created by bulk_invoke().
/// 
/// Execution categories describe the ordering of these function invocations with respect to one another.
///
/// 1. sequenced: Invocations are sequenced.
/// 2. concurrent: Unblocked invocations make progress.
/// 3. parallel: When invocations occur on the same thread, they are sequenced. When invocations occur on
///              different threads, they are unsequenced.
/// 4. unsequenced: Invocations are unsequenced.
///
/// Execution categories are represented in the C++ type system using tag types. Different components of
/// Agency use these types to reason about forward progress guarantees and validate that forward progress
/// guarantees satisfy forward progress requirements. 

#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>

#include <type_traits>
#include <stddef.h>

namespace agency
{


/// \brief Type representing the sequenced execution category.
/// \ingroup execution_categories
struct sequenced_execution_tag {};

/// \brief Type representing the concurrent execution category.
/// \ingroup execution_categories
struct concurrent_execution_tag {};

/// \brief Type representing the parallel execution category.
/// \ingroup execution_categories
struct parallel_execution_tag {};

/// \brief Type representing the unsequenced execution category.
/// \ingroup execution_categories
struct unsequenced_execution_tag {};

/// \brief Type representing the scoped execution category.
/// \ingroup execution_categories
template<class ExecutionCategory1, class ExecutionCategory2>
struct scoped_execution_tag
{
  using outer_execution_category = ExecutionCategory1;
  using inner_execution_category = ExecutionCategory2;
};

/// \brief Type representing the dynamic execution category.
/// \ingroup execution_categories
struct dynamic_execution_tag {};


namespace detail
{


// XXX maybe "weakness" is the wrong way to describe what we're really interested in here
//     we just want some idea of substitutability

//  "<" means "is weaker than"
//  "<" is transitive
// if category A is weaker than category B,
// then agents in category A can be executed with agents in category B
//
// these relationships should be true
//
// parallel_execution_tag    < sequenced_execution_tag
// parallel_execution_tag    < concurrent_execution_tag
// unsequenced_execution_tag < parallel_execution_tag
// dynamic_execution_tag     < unsequenced_execution_tag
//
// XXX figure out how scoped_execution_tag sorts

// in general, categories are not weaker than another
template<class ExecutionCategory1, class ExecutionCategory2>
struct is_weaker_than : std::false_type {};

// all categories are weaker than themselves
template<class ExecutionCategory>
struct is_weaker_than<ExecutionCategory,ExecutionCategory> : std::true_type {};

// dynamic is weaker than everything else
template<class ExecutionCategory2>
struct is_weaker_than<dynamic_execution_tag, ExecutionCategory2> : std::true_type {};

// introduce this specialization to disambiguate two other specializations
template<>
struct is_weaker_than<dynamic_execution_tag, dynamic_execution_tag> : std::true_type {};

// unsequenced is weaker than everything except dynamic
template<class ExecutionCategory2>
struct is_weaker_than<unsequenced_execution_tag, ExecutionCategory2> : std::true_type {};

// introduce this specialization to disambiguate two other specializations
template<>
struct is_weaker_than<unsequenced_execution_tag, unsequenced_execution_tag> : std::true_type {};

template<>
struct is_weaker_than<unsequenced_execution_tag, dynamic_execution_tag> : std::false_type {};

// parallel is weaker than sequenced & concurrent
template<>
struct is_weaker_than<parallel_execution_tag, sequenced_execution_tag> : std::true_type {};

template<>
struct is_weaker_than<parallel_execution_tag, concurrent_execution_tag> : std::true_type {};


template<class ExecutionCategory>
struct is_scoped_execution_category : std::false_type {};


template<class ExecutionCategory1, class ExecutionCategory2>
struct is_scoped_execution_category<scoped_execution_tag<ExecutionCategory1,ExecutionCategory2>> : std::true_type {};


template<class ExecutionCategory>
struct execution_depth : std::integral_constant<size_t, 1> {};


template<class ExecutionCategory1, class ExecutionCategory2>
struct execution_depth<scoped_execution_tag<ExecutionCategory1,ExecutionCategory2>>
  : std::integral_constant<
      size_t,
      1 + execution_depth<ExecutionCategory2>::value
    >
{};


// this namespace contains the implementation of common_execution_category
namespace common_execution_category_detail
{


// because the implementation of common_execution_category2 is recursive, we introduce
// a forward declaration so it may call itself.
template<class ExecutionCategory1, class ExecutionCategory2>
struct common_execution_category2;

template<class ExecutionCategory1, class ExecutionCategory2>
using common_execution_category2_t = typename common_execution_category2<ExecutionCategory1,ExecutionCategory2>::type;


// the implementation of common_execution_category2_impl is recursive, and there are two base cases

// base case 1: the input categories have different depths
// i.e., one may be "flat" and the other scoped,
// or both may be scoped but have different depths
template<class ExecutionCategory1, class ExecutionCategory2, size_t depth1, size_t depth2>
struct common_execution_category2_impl
{
  // there's no commonality between the two input categories
  // so the result is "dynamic" -- there are no static guarantees that can be provided
  using type = dynamic_execution_tag;
};


// base case 2: both input categories are "flat"
template<class ExecutionCategory1, class ExecutionCategory2>
struct common_execution_category2_impl<ExecutionCategory1,ExecutionCategory2,1,1>
{
  // both ExecutionCategory1 & ExecutionCategory2 have depth 1 -- they are "flat"

  // if one of the two categories is weaker than the other, then return it
  // otherwise, return the weakest static guarantee: unsequenced
  using type = conditional_t<
    is_weaker_than<ExecutionCategory1,ExecutionCategory2>::value,
    ExecutionCategory1,
    conditional_t<
      is_weaker_than<ExecutionCategory2,ExecutionCategory1>::value,
      ExecutionCategory2,
      unsequenced_execution_tag
    >
  >;
};


// recursive case: both input categories are scoped
template<class OuterCategory1, class InnerCategory1, class OuterCategory2, class InnerCategory2, size_t depth>
struct common_execution_category2_impl<
  scoped_execution_tag<OuterCategory1,InnerCategory1>,
  scoped_execution_tag<OuterCategory2,InnerCategory2>,
  depth,depth
>
{
  // both categories are scoped and they have the same depth
  // XXX it may not matter so much that the depth is the same.
  //     we may still be able to apply this recipe sensibly even
  //     when the two input categories' depths differ

  // the result is scoped. apply common_execution_category to
  // the inputs' constituents to get the result's constituents
  using type = scoped_execution_tag<
    common_execution_category2_t<OuterCategory1,OuterCategory2>,
    common_execution_category2_t<InnerCategory1,InnerCategory2>
  >;
};


template<class ExecutionCategory1, class ExecutionCategory2>
struct common_execution_category2
{
  using type = typename common_execution_category2_impl<
    ExecutionCategory1,
    ExecutionCategory2,
    execution_depth<ExecutionCategory1>::value,
    execution_depth<ExecutionCategory2>::value
  >::type;
};


} // end common_execution_category_detail


// common_execution_category is a type trait which, given one or more possibly different execution categories,
// returns a category representing the strongest guarantees that can be made given the different input possibilities
template<class ExecutionCategory, class... ExecutionCategories>
struct common_execution_category;

template<class ExecutionCategory, class... ExecutionCategories>
using common_execution_category_t = typename common_execution_category<ExecutionCategory,ExecutionCategories...>::type;


// the implementation of common_execution_category is recursive
// this is the recursive case
template<class ExecutionCategory1, class ExecutionCategory2, class... ExecutionCategories>
struct common_execution_category<ExecutionCategory1, ExecutionCategory2, ExecutionCategories...>
{
  using type = common_execution_category_t<
    ExecutionCategory1,
    common_execution_category_t<ExecutionCategory2, ExecutionCategories...>
  >;
};

// base case 1: a single category
template<class ExecutionCategory>
struct common_execution_category<ExecutionCategory>
{
  using type = ExecutionCategory;
};

// base case 2: two categories
template<class ExecutionCategory1, class ExecutionCategory2>
struct common_execution_category<ExecutionCategory1,ExecutionCategory2>
{
  // with two categories, we lower onto the two category
  // implementation inside common_execution_category_detail
  using type = common_execution_category_detail::common_execution_category2_t<ExecutionCategory1,ExecutionCategory2>;
};


} // end detail
} // end agency

