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


template<class ExecutionCategory1, class ExecutionCategory2>
struct scoped_execution_tag
{
  using outer_execution_category = ExecutionCategory1;
  using inner_execution_category = ExecutionCategory2;
};


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
//
// XXX figure out how scoped_execution_tag sorts

// most categories are not weaker than another
template<class ExecutionCategory1, class ExecutionCategory2>
struct is_weaker_than : std::false_type {};

// all categories are weaker than themselves
template<class ExecutionCategory>
struct is_weaker_than<ExecutionCategory,ExecutionCategory> : std::true_type {};

// unsequenced is weaker than everything
template<class ExecutionCategory2>
struct is_weaker_than<unsequenced_execution_tag, ExecutionCategory2> : std::true_type {};

// introduce this specialization to disambiguate two other specializations
template<>
struct is_weaker_than<unsequenced_execution_tag, unsequenced_execution_tag> : std::true_type {};

// parallel is weaker than everything except unsequenced
template<class ExecutionCategory2>
struct is_weaker_than<parallel_execution_tag, ExecutionCategory2> : std::true_type {};

// introduce this specialization to disambiguate two other specializations
template<>
struct is_weaker_than<parallel_execution_tag, parallel_execution_tag> : std::true_type {};

template<>
struct is_weaker_than<parallel_execution_tag, unsequenced_execution_tag> : std::false_type {};


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


} // end detail
} // end agency

