#include <agency/sequential_executor.hpp>
#include <agency/nested_executor.hpp>
#include <agency/executor_traits.hpp>
#include <agency/execution_agent.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/index_cast.hpp>
#include <agency/coordinate.hpp>
#include <agency/execution_policy.hpp>
#include <iostream>
#include <cassert>
#include <functional>


template<class Tuple,
         class Enable = typename std::enable_if<
           (std::tuple_size<typename std::decay<Tuple>::type>::value == 0)
         >::type
        >
void __tuple_find_if_impl(Tuple&& filtered_tuple) {}


template<class Tuple,
         class Enable = typename std::enable_if<
           (std::tuple_size<typename std::decay<Tuple>::type>::value != 0)
         >::type
        >
auto __tuple_find_if_impl(Tuple&& filtered_tuple)
  -> decltype(
       agency::detail::get<0>(std::forward<Tuple>(filtered_tuple))
     )
{
  return agency::detail::get<0>(std::forward<Tuple>(filtered_tuple));
}


template<template<class> class MetaFunction, class Tuple>
TUPLE_UTILITY_ANNOTATION
auto tuple_find_if(Tuple&& t)
  -> decltype(
       __tuple_find_if_impl(__tu::tuple_filter_invoke<MetaFunction>(std::forward<Tuple>(t), agency::detail::forwarder{}))
     )
{
  return __tuple_find_if_impl(__tu::tuple_filter_invoke<MetaFunction>(std::forward<Tuple>(t), agency::detail::forwarder{}));
}


struct null_type {};

std::ostream& operator<<(std::ostream& os, null_type)
{
  return os << "null";
}


// when looking for null_type, strip references first
template<class T>
struct is_null : std::is_same<null_type, typename std::decay<T>::type> {};


template<class T>
struct is_not_null : 
  std::integral_constant<
    bool,
    !is_null<T>::value
  >
{};



template<class Tuple>
auto tuple_find_non_null(Tuple&& t)
  -> decltype(tuple_find_if<is_not_null>(std::forward<Tuple>(t)))
{
  return tuple_find_if<is_not_null>(std::forward<Tuple>(t));
}


template<size_t level, class T, class... Args>
struct shared_parameter
{
  T make(std::integral_constant<size_t,level>) const
  {
    return __tu::make_from_tuple<T>(args_);
  }

  template<size_t other_level>
  null_type make(std::integral_constant<size_t,other_level>) const
  {
    return null_type{};
  }

  agency::detail::tuple<Args...> args_;
};

template<class T> struct is_shared_parameter : std::false_type {};
template<size_t level, class T, class... Args>
struct is_shared_parameter<shared_parameter<level,T,Args...>> : std::true_type {};

template<class T>
struct is_shared_parameter_ref
  : std::integral_constant<
      bool,
      (std::is_reference<T>::value && is_shared_parameter<typename std::remove_reference<T>::type>::value)
    >
{};

template<size_t level, class T, class... Args>
shared_parameter<level, T,Args...> share(Args&&... args)
{
  return shared_parameter<level, T,Args...>{std::make_tuple(std::forward<Args>(args)...)};
}

template<size_t level, class T>
shared_parameter<level,T,T> share(const T& val)
{
  return shared_parameter<level,T,T>{std::make_tuple(val)};
}


template<size_t level>
struct call_make
{
  template<size_t other_level, class T, class... Args>
  auto operator()(const shared_parameter<other_level,T,Args...>& parm) const
    -> decltype(
         parm.make(std::integral_constant<size_t,level>{})
       )
  {
    return parm.make(std::integral_constant<size_t,level>{});
  }
};


template<size_t column_idx, size_t... row_idx, class TupleMatrix>
auto tuple_matrix_column_impl(agency::detail::index_sequence<row_idx...>, TupleMatrix&& mtx)
  -> decltype(
       agency::detail::tie(
           agency::detail::get<column_idx>(agency::detail::get<row_idx>(std::forward<TupleMatrix>(mtx)))...
         )
     )
{
  return agency::detail::tie(
    agency::detail::get<column_idx>(agency::detail::get<row_idx>(std::forward<TupleMatrix>(mtx)))...
  );
}


template<size_t column_idx, class TupleMatrix>
auto tuple_matrix_column(TupleMatrix&& mtx)
  -> decltype(
       tuple_matrix_column_impl<column_idx>(
         agency::detail::make_index_sequence<
           std::tuple_size<
             typename std::decay<TupleMatrix>::type
           >::value
         >{},
         std::forward<TupleMatrix>(mtx)
       )
     )
{
  return tuple_matrix_column_impl<column_idx>(
    agency::detail::make_index_sequence<
      std::tuple_size<
        typename std::decay<TupleMatrix>::type
      >::value
    >{},
    std::forward<TupleMatrix>(mtx)
  );
}


template<size_t... RowIndex, class SharedArgTuple>
auto make_shared_parameter_matrix_impl(agency::detail::index_sequence<RowIndex...>,
                                       SharedArgTuple&& shared_arg_tuple)
  -> decltype(
       agency::detail::make_tuple(
         __tu::tuple_map(call_make<RowIndex>{}, std::forward<SharedArgTuple>(shared_arg_tuple))...
       )
     )
{
  return agency::detail::make_tuple(
    __tu::tuple_map(call_make<RowIndex>{}, std::forward<SharedArgTuple>(shared_arg_tuple))...
  );
}


// create a shared parameter matrix
// the rows correspond to levels of the executor's hierarchy
// the columns correspond to shared arguments
template<size_t num_rows, class SharedArgTuple>
auto make_shared_parameter_matrix(SharedArgTuple&& shared_arg_tuple)
  -> decltype(
       make_shared_parameter_matrix_impl(
         agency::detail::make_index_sequence<num_rows>{},
         std::forward<SharedArgTuple>(shared_arg_tuple)
       )
     )
{
  return make_shared_parameter_matrix_impl(
    agency::detail::make_index_sequence<num_rows>{},
    std::forward<SharedArgTuple>(shared_arg_tuple)
  );
}


template<size_t... RowIndex, class TupleMatrix>
auto extract_shared_parameters_from_rows_impl(agency::detail::index_sequence<RowIndex...>, TupleMatrix&& mtx)
  -> decltype(
       agency::detail::tie(
         tuple_find_non_null(agency::detail::get<RowIndex>(std::forward<TupleMatrix>(mtx)))...
       )
     )
{
  return agency::detail::tie(
    tuple_find_non_null(agency::detail::get<RowIndex>(mtx))...
  );
}


template<class TupleMatrix>
auto extract_shared_parameters_from_rows(TupleMatrix&& mtx)
  -> decltype(
       extract_shared_parameters_from_rows_impl(
         agency::detail::make_index_sequence<
           std::tuple_size<
             typename std::decay<TupleMatrix>::type
           >::value
         >{},
         std::forward<TupleMatrix>(mtx)
       )
     )
{
  return extract_shared_parameters_from_rows_impl(
    agency::detail::make_index_sequence<
      std::tuple_size<
        typename std::decay<TupleMatrix>::type
      >::value
    >{},
    std::forward<TupleMatrix>(mtx)
  );
}


template<size_t... ColumnIndex, class TupleMatrix>
auto make_transposed_view_impl(agency::detail::index_sequence<ColumnIndex...>, TupleMatrix&& mtx)
  -> decltype(
       agency::detail::make_tuple(
         tuple_matrix_column<ColumnIndex>(std::forward<TupleMatrix>(mtx))...
       )
     )
{
  // to create a transposed view of the matrix, create a tuple of the columns
  return agency::detail::make_tuple(
    tuple_matrix_column<ColumnIndex>(std::forward<TupleMatrix>(mtx))...
  );
}


template<size_t num_columns, class TupleMatrix>
auto make_transposed_view(TupleMatrix&& mtx)
  -> decltype(
       make_transposed_view_impl(
         agency::detail::make_index_sequence<num_columns>{},
         std::forward<TupleMatrix>(mtx)
       )
     )
{
  return make_transposed_view_impl(
    agency::detail::make_index_sequence<num_columns>{},
    std::forward<TupleMatrix>(mtx)
  );
}


// a TupleMatrix is a tuple of tuples
// where each nested tuple is the same type
template<class TupleMatrix>
struct tuple_matrix_shape
{
  static const size_t rows = std::tuple_size<TupleMatrix>::value;

  using row_type = typename std::decay<
    typename std::tuple_element<0,TupleMatrix>::type
  >::type;

  static const size_t columns = std::tuple_size<row_type>::value;
};



// the number of shared parameters is equal to the number of columns of TupleMatrix
// this is equal to the tuple_size of tuple_element 0
template<class TupleMatrix,
         size_t NumSharedParams = tuple_matrix_shape<typename std::decay<TupleMatrix>::type>::columns
        >
auto unpack_shared_params(TupleMatrix&& shared_param_matrix)
  -> decltype(
       extract_shared_parameters_from_rows(
         make_transposed_view<NumSharedParams>(
           std::forward<TupleMatrix>(shared_param_matrix)
         )
       )
     )
{
  // to transform the shared_param_matrix into a tuple of shared parameters
  // we need to find the actual (non-null) parameter in each column of the matrix
  // the easiest way to do this is to tranpose the matrix and extract the shared parameter from each row of the transpose
  return extract_shared_parameters_from_rows(
    make_transposed_view<NumSharedParams>(
      std::forward<TupleMatrix>(shared_param_matrix)
    )
  );
}


template<class Executor, class Function, class SharedArgTuple>
void bulk_invoke_executor_impl(Executor& exec, Function f, typename agency::executor_traits<Executor>::shape_type shape, SharedArgTuple&& shared_arg_tuple)
{
  using traits = agency::executor_traits<Executor>;

  // explicitly construct the shared parameter
  // it gets copy constructed by the executor
  // XXX problems with this approach
  //     1. the type of the shared parameter is constructed twice
  //       1.1 we can ameliorate this if executors receive shared parameters as forwarding references
  //           and move them into exec.bulk_invoke()
  //     2. requires the type of the shared parameter to be copy constructable
  //       2.1 we can fix this if executors receive shared parameters as forwarding references
  //     3. won't be able to support concurrent construction

  // turn the tuple of shared arguments into a tuple of shared initializers by invoking .make()

  // create a shared parameter matrix
  // the rows correspond to levels of the executor's hierarchy
  // the columns correspond to shared arguments
  const size_t executor_depth = agency::detail::execution_depth<
    typename traits::execution_category
  >::value;

  auto shared_init = make_shared_parameter_matrix<executor_depth>(std::forward<SharedArgTuple>(shared_arg_tuple));

  using shared_param_type = typename traits::template shared_param_type<decltype(shared_init)>;

  traits::bulk_invoke(exec, [=](typename traits::index_type idx, shared_param_type& shared_param_matrix)
  {
    auto shared_params = unpack_shared_params(shared_param_matrix);

    f(idx, shared_params);
  },
  shape,
  shared_init);
}


template<size_t I, class Arg>
typename std::enable_if<
  !is_shared_parameter<typename std::decay<Arg>::type>::value,
  Arg&&
>::type
  hold_shared_parameters_place(Arg&& arg)
{
  return std::forward<Arg>(arg);
}


using placeholder_tuple_t = agency::detail::tuple<
  decltype(std::placeholders::_1),
  decltype(std::placeholders::_2),
  decltype(std::placeholders::_3),
  decltype(std::placeholders::_4),
  decltype(std::placeholders::_5),
  decltype(std::placeholders::_6),
  decltype(std::placeholders::_7),
  decltype(std::placeholders::_8),
  decltype(std::placeholders::_9),
  decltype(std::placeholders::_10)
>;


template<size_t i>
using placeholder = typename std::tuple_element<i, placeholder_tuple_t>::type;


template<size_t I, class T>
typename std::enable_if<
  is_shared_parameter<typename std::decay<T>::type>::value,
  placeholder<I>
>::type
  hold_shared_parameters_place(T&&)
{
  return placeholder<I>{};
}


struct forwarder
{
  template<class... Args>
  auto operator()(Args&&... args) const
    -> decltype(agency::detail::forward_as_tuple(args...))
  {
    return agency::detail::forward_as_tuple(args...);
  }
};


template<class... Args>
auto forward_shared_parameters_as_tuple(Args&&... args)
  -> decltype(
       __tu::tuple_filter_invoke<is_shared_parameter_ref>(
         agency::detail::forward_as_tuple(std::forward<Args>(args)...),
         forwarder{}
       )
     )
{
  auto t = agency::detail::forward_as_tuple(std::forward<Args>(args)...);

  return __tu::tuple_filter_invoke<is_shared_parameter_ref>(t, forwarder{});
}


template<class Function>
struct unpack_shared_args_and_invoke
{
  mutable Function f_;

  template<class Index, class Tuple>
  void operator()(Index&& idx, Tuple&& shared_args) const
  {
    // create one big tuple of the arguments so we can just call tuple_apply
    auto args = __tu::tuple_prepend_invoke(std::forward<Tuple>(shared_args), std::forward<Index>(idx), forwarder{});

    __tu::tuple_apply(f_, args);
  }
};


// J... is a bit vector indicating which of args is a shared parameter
// then I... is the exclusive scan of J
template<size_t... I, class Function, class... Args>
auto bind_unshared_parameters_impl(agency::detail::index_sequence<I...>, Function f, Args&&... args)
  -> decltype(
       std::bind(f, hold_shared_parameters_place<1 + I>(std::forward<Args>(args))...)
     )
{
  // we add 1 to I to account for the executor_idx argument passed by unpack_shared_args_and_invoke
  // above as the first parameter to f
  return std::bind(f, hold_shared_parameters_place<1 + I>(std::forward<Args>(args))...);
}


template<class... Args>
struct arg_is_shared
{
  using tuple_type = agency::detail::tuple<Args...>;

  template<size_t i>
  using map = is_shared_parameter<
    typename std::decay<
      typename std::tuple_element<i,tuple_type>::type
    >::type
  >;
};


// we need to generate
template<class... Args>
using scanned_shared_argument_indices = agency::detail::transform_exclusive_scan_index_sequence<
  arg_is_shared<Args...>::template map,
  0,
  agency::detail::index_sequence_for<Args...>
>;

template<class Function, class... Args>
auto bind_unshared_parameters(Function f, Args&&... args)
  -> decltype(
       bind_unshared_parameters_impl(scanned_shared_argument_indices<Args...>{}, f, std::forward<Args>(args)...)
     )
{
  return bind_unshared_parameters_impl(scanned_shared_argument_indices<Args...>{}, f, std::forward<Args>(args)...);
}


template<class Executor, class Function, class... Args>
void bulk_invoke_executor(Executor&& exec, Function f, typename agency::executor_traits<typename std::decay<Executor>::type>::shape_type shape, Args&&... args)
{
  // the _1 is for the executor idx parameter, which is the first parameter passed to f
  auto g = bind_unshared_parameters(f, std::placeholders::_1, std::forward<Args>(args)...);

  // make a tuple of the shared args
  auto shared_arg_tuple = forward_shared_parameters_as_tuple(std::forward<Args>(args)...);

  // create h which takes a tuple of args and calls g
  auto h = unpack_shared_args_and_invoke<decltype(g)>{g};

  bulk_invoke_executor_impl(exec, h, shape, shared_arg_tuple);
}


size_t rank(agency::size2 shape, agency::size2 idx)
{
  return agency::get<1>(shape) * agency::get<0>(idx) + agency::get<1>(idx);
}

size_t rank(agency::size3 shape, agency::size3 idx)
{
  agency::size2 idx2{agency::get<0>(idx), agency::get<1>(idx)};
  agency::size2 shape2{agency::get<0>(shape), agency::get<1>(shape)};

  auto rank2 = rank(shape2, idx2);

  return agency::get<2>(idx) + agency::get<2>(shape) * rank2;
}

void test1()
{
  using executor_type1 = agency::nested_executor<agency::sequential_executor,agency::sequential_executor>;

  using executor_type2 = agency::nested_executor<agency::sequential_executor,executor_type1>;

  executor_type2 exec;
  executor_type2::shape_type shape{2,2,2};

  auto lambda = [=](executor_type2::index_type idx, int& outer_shared, int& middle_shared, int& inner_shared)
  {
    std::cout << "idx: " << idx << std::endl;
    std::cout << "outer_shared: " << outer_shared << std::endl;
    std::cout << "middle_shared: " << middle_shared << std::endl;
    std::cout << "inner_shared:  " << inner_shared << std::endl;

    auto outer_shape_ = shape;
    agency::size3 outer_shape{std::get<0>(outer_shape_), std::get<1>(outer_shape_), std::get<2>(outer_shape_)};
    auto outer_idx_   = idx;
    agency::size3 outer_idx{std::get<0>(outer_idx_), std::get<1>(outer_idx_), std::get<2>(outer_idx_)};

    assert(outer_shared  == rank(outer_shape, outer_idx) + 1);


    auto middle_shape_ = agency::detail::tuple_tail(shape);
    agency::size2 middle_shape{std::get<0>(middle_shape_), std::get<1>(middle_shape_)};
    auto middle_idx_   = agency::detail::tuple_tail(idx);
    agency::size2 middle_idx{std::get<0>(middle_idx_), std::get<1>(middle_idx_)};

    assert(middle_shared == rank(middle_shape, middle_idx) + 2);

    auto inner_shape  = agency::detail::tuple_tail(middle_shape);
    auto inner_idx    = agency::detail::tuple_tail(middle_idx);

    assert(inner_shared  == agency::detail::get<0>(inner_idx) + 3);

    ++outer_shared;
    ++middle_shared;
    ++inner_shared;
  };

  bulk_invoke_executor(exec, lambda, shape, share<0>(1), share<1>(2), share<2>(3));
}


template<class ExecutorTraits, class AgentTraits, class Function, size_t... UserArgIndices>
struct execute_agent_functor
{
  using agent_type        = typename AgentTraits::execution_agent_type;
  using agent_param_type  = typename AgentTraits::param_type;
  using agent_domain_type = typename AgentTraits::domain_type;
  using agent_shape_type  = decltype(std::declval<agent_domain_type>().shape());

  using executor_shape_type = typename ExecutorTraits::shape_type;

  agent_param_type    agent_param_;
  agent_shape_type    agent_shape_;
  executor_shape_type executor_shape_;
  Function            f_;

  using agent_index_type    = typename AgentTraits::index_type;
  using executor_index_type = typename ExecutorTraits::index_type;

  template<class... Args>
  void operator()(const executor_index_type& executor_idx, Args&&... args)
  {
    // collect all parameters into a tuple of references
    auto args_tuple = agency::detail::forward_as_tuple(std::forward<Args>(args)...);

    // split the parameters into user parameters and agent parameters
    auto user_args         = agency::detail::tuple_take_view<sizeof...(UserArgIndices)>(args_tuple);
    auto agent_shared_args = agency::detail::tuple_drop_view<sizeof...(UserArgIndices)>(args_tuple);

    // turn the executor index into an agent index
    using agent_index_type = typename AgentTraits::index_type;
    auto agent_idx = agency::detail::index_cast<agent_index_type>(executor_idx, executor_shape_, agent_shape_);

    // AgentTraits::execute expects a function whose only parameter is agent_type
    // so we have to 
    auto invoke_f = [&user_args,this](agent_type& self)
    {
      // invoke f by passing the agent, then the user's parameters
      f_(self, agency::detail::get<UserArgIndices>(user_args)...);
    };

    AgentTraits::execute(invoke_f, agent_idx, agent_param_, agent_shared_args);
  }
};


template<size_t... UserArgIndices, size_t... SharedArgIndices, class ExecutionPolicy, class Function, class... Args>
void bulk_invoke_new_impl(agency::detail::index_sequence<UserArgIndices...>,
                          agency::detail::index_sequence<SharedArgIndices...>,
                          ExecutionPolicy& policy, Function f, Args&&... args)
{
  using agent_type = agency::sequential_group<agency::sequential_agent>;
  using agent_traits = agency::execution_agent_traits<agent_type>;

  // get the parameters of the agent
  auto param = policy.param();
  auto agent_shape = agent_traits::domain(param).shape();
  auto agent_shared_param_tuple = agent_traits::make_shared_param_tuple(param);

  using executor_type = typename ExecutionPolicy::executor_type;
  using executor_traits = agency::executor_traits<executor_type>;

  // convert the shape of the agent into the type of the executor's shape
  using executor_shape_type = typename executor_traits::shape_type;
  executor_shape_type executor_shape = agency::detail::shape_cast<executor_shape_type>(agent_shape);

  // create the function that will marshal parameters received from bulk_invoke(executor) and execute the agent
  auto lambda = execute_agent_functor<executor_traits,agent_traits,Function,UserArgIndices...>{param, agent_shape, executor_shape, f};

  bulk_invoke_executor(policy.executor(), lambda, executor_shape, std::forward<Args>(args)..., share<SharedArgIndices>(std::get<SharedArgIndices>(agent_shared_param_tuple))...);
}


template<class ExecutionPolicy, class Function, class... Args>
void bulk_invoke_new(ExecutionPolicy& policy, Function f, Args&&... args)
{
  bulk_invoke_new_impl(agency::detail::index_sequence_for<Args...>(), agency::detail::make_index_sequence<2>(), policy, f, std::forward<Args>(args)...);
}

void test2()
{
  auto lambda = [](agency::sequential_group<agency::sequential_agent>& self, int local_param)
  {
    std::cout << "index: " << self.index() << " local_param: " << local_param << std::endl;
  };

  auto policy = agency::seq(2,agency::seq(2));
  bulk_invoke_new(policy, lambda, 13);
}


int main()
{
//  test1();
  test2();

  return 0;
}

