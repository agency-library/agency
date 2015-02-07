#pragma once

#include <type_traits>
#include <stddef.h>

namespace agency
{


// XXX consider whether there should be derived-from relationships between these
struct sequential_execution_tag {};
struct concurrent_execution_tag {};
struct parallel_execution_tag {};
struct vector_execution_tag {};


template<class ExecutionCategory1, class ExecutionCategory2>
struct nested_execution_tag
{
  using outer_execution_category = ExecutionCategory1;
  using inner_execution_category = ExecutionCategory2;
};


// XXX need some way to compare the strength of these at compile time
// the following 

//  "<" means "is weaker than"
//  "<" is transitive
// if category A is weaker than category B,
// then agents in category A can be executed with agents in category B
//
// these relationships should be true
//
// parallel_execution_tag < sequential_execution_tag
// parallel_execution_tag < concurrent_execution_tag
// vector_execution_tag   < parallel_execution_tag
//
// XXX figure out how sequential is related to concurrent
//
// XXX figure out how nested_execution_tag sorts


namespace detail
{


template<class ExecutionCategory>
struct is_nested_execution_category : std::false_type {};


template<class ExecutionCategory1, class ExecutionCategory2>
struct is_nested_execution_category<nested_execution_tag<ExecutionCategory1,ExecutionCategory2>> : std::true_type {};


template<class ExecutionCategory>
struct execution_depth : std::integral_constant<size_t, 1> {};


template<class ExecutionCategory1, class ExecutionCategory2>
struct execution_depth<nested_execution_tag<ExecutionCategory1,ExecutionCategory2>>
  : std::integral_constant<
      size_t,
      1 + execution_depth<ExecutionCategory2>::value
    >
{};


} // end detail
} // end agency

