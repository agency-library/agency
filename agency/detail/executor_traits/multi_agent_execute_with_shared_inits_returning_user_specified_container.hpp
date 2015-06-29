#pragma once

#include <agency/detail/config.hpp>
#include <agency/new_executor_traits.hpp>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{
namespace multi_agent_execute_with_shared_inits_returning_user_specified_container_implementation_strategies
{


struct use_multi_agent_execute_with_shared_inits_returning_user_specified_container_member_function {};

// XXX struct use_multi_agent_execute_with_shared_inits_returning_user_specified_container_member_function {};

struct use_multi_agent_execute_returning_user_specified_container {};


template<class Container, class Executor, class Function, class T1, class... Types>
using select_multi_agent_execute_with_shared_inits_returning_user_specified_container_implementation =
  typename std::conditional<
    has_multi_agent_execute_with_shared_inits_returning_user_specified_container<
      Container,
      Executor,
      Function,
      T1,
      Types...
    >::value,
    use_multi_agent_execute_with_shared_inits_returning_user_specified_container_member_function,
    use_multi_agent_execute_returning_user_specified_container
  >::type;


template<class Container, class Executor, class Function, class T1, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_execute_with_shared_inits_returning_user_specified_container_member_function,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   T1&& outer_shared_init, Types&&... inner_shared_inits)
{
  return ex.template execute<Container>(f, shape, std::forward<T1>(outer_shared_init), std::forward<Types>(inner_shared_inits)...);
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


template<class Result, class Function, class Shape, class TupleOfContainers>
struct multi_agent_execute_with_shared_inits_functor
{
  mutable Function f;
  Shape shape;
  TupleOfContainers& shared_arg_containers;

  template<size_t depth, class AgentIndex>
  __AGENCY_ANNOTATION
  size_t rank_in_group(const AgentIndex& idx) const
  {
    // to compute the rank of an index at a particular depth,
    // first prepend 0 (1) to idx (shape) to represent an index of the root group (it has none otherwise)
    // XXX seems like index_cast() should just do the right thing for empty indices
    //     it would correspond to a single-agent task
    auto augmented_idx   = detail::tuple_prepend(detail::wrap_scalar(idx), size_t{0});
    auto augmented_shape = detail::tuple_prepend(detail::wrap_scalar(shape), size_t{1});
    
    // take the first depth+1 (plus one because we prepended 1) indices of the index & shape and do an index_cast to size_t
    return detail::index_cast<size_t>(detail::tuple_take<depth+1>(augmented_idx),
                                      detail::tuple_take<depth+1>(augmented_shape),
                                      detail::shape_size(detail::tuple_take<depth+1>(augmented_shape)));
  }

  template<size_t... ContainerIndices, class AgentIndex>
  __AGENCY_ANNOTATION
  Result impl(detail::index_sequence<ContainerIndices...>, AgentIndex&& agent_idx) const
  {
    return f(std::forward<AgentIndex>(agent_idx),                                                      // pass the agent index
      std::get<ContainerIndices>(shared_arg_containers)[rank_in_group<ContainerIndices>(agent_idx)]... // pass the arguments coming in from shared parameters
    );
  }

  template<class Index>
  __AGENCY_ANNOTATION
  Result operator()(Index&& idx) const
  {
    static const size_t num_containers = std::tuple_size<TupleOfContainers>::value;
    return impl(detail::make_index_sequence<num_containers>(), std::forward<Index>(idx));
  }
};


template<class Result, class Function, class Shape, class TupleOfContainers>
__AGENCY_ANNOTATION
multi_agent_execute_with_shared_inits_functor<Result,Function,Shape,TupleOfContainers>
  make_multi_agent_execute_with_shared_inits_functor(Function f, Shape shape, TupleOfContainers& tuple_of_containers)
{
  return multi_agent_execute_with_shared_inits_functor<Result,Function,Shape,TupleOfContainers>{f,shape,tuple_of_containers};
} 


template<class Container, class Executor, class Function, class... Types>
Container multi_agent_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_execute_returning_user_specified_container,
                                                                                   Executor& ex,
                                                                                   Function f,
                                                                                   typename new_executor_traits<Executor>::shape_type shape,
                                                                                   Types&&... shared_inits)
{
  // create a tuple of containers holding a shared parameter for each group
  auto shared_param_containers_tuple = make_tuple_of_shared_parameter_containers(ex, shape, std::forward<Types>(shared_inits)...);

  using index_type = typename new_executor_traits<Executor>::index_type;
  using result_type = typename std::result_of<
    Function(
      index_type,
      typename std::decay<Types>::type&...
    )
  >::type;

  // wrap f with a functor to map container elements to shared parameters
  auto g = make_multi_agent_execute_with_shared_inits_functor<result_type>(f, shape, shared_param_containers_tuple);

  return new_executor_traits<Executor>::template execute<Container>(ex, g, shape);
} // end multi_agent_execute_with_shared_inits_returning_user_specified_container()


} // end multi_agent_execute_with_shared_inits_returning_user_specified_container_implementation_strategies
} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Container, class Function, class T1, class... Types,
           class Enable>
Container new_executor_traits<Executor>
  ::execute(typename new_executor_traits<Executor>::executor_type& ex,
            Function f,
            typename new_executor_traits<Executor>::shape_type shape,
            T1&& outer_shared_init, Types&&... inner_shared_inits)
{
  namespace ns = detail::new_executor_traits_detail::multi_agent_execute_with_shared_inits_returning_user_specified_container_implementation_strategies;

  using implementation_strategy = ns::select_multi_agent_execute_with_shared_inits_returning_user_specified_container_implementation<
    Container,
    Executor,
    Function,
    T1&&, Types&&...
  >;

  return ns::multi_agent_execute_with_shared_inits_returning_user_specified_container<Container>(implementation_strategy(), ex, f, shape, std::forward<T1>(outer_shared_init), std::forward<Types>(inner_shared_inits)...);
} // end new_executor_traits::execute()


} // end agency

