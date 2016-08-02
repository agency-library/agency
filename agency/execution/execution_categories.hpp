#pragma once

#include <type_traits>
#include <stddef.h>

namespace agency
{


// XXX consider whether there should be derived-from relationships between these
struct sequential_execution_tag {};
struct concurrent_execution_tag {};
struct parallel_execution_tag {};
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
// parallel_execution_tag    < sequential_execution_tag
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

