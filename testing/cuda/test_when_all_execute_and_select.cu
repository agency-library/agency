#include <agency/cuda/grid_executor.hpp>
#include <agency/cuda/detail/kernel.hpp>
#include <memory>

template<class Function, class IndexFunction, class OuterArgumentPointer, class InnerFactory, class... DependencyPointers>
struct when_all_execute_functor
{
  Function                                     f_;
  IndexFunction                                index_function_;
  agency::detail::tuple<DependencyPointers...> dependency_ptrs_;
  OuterArgumentPointer                         outer_arg_ptr_;
  InnerFactory                                 inner_factory_;

  __host__ __device__
  when_all_execute_functor(Function f, IndexFunction index_function, OuterArgumentPointer outer_arg_ptr, InnerFactory inner_factory, DependencyPointers... dependency_ptrs)
    : f_(f),
      index_function_(index_function),
      dependency_ptrs_(dependency_ptrs...),
      outer_arg_ptr_(outer_arg_ptr),
      inner_factory_(inner_factory)
  {}

  template<size_t... Indices>
  __device__
  void impl(agency::detail::index_sequence<Indices...>)
  {
    auto idx = index_function_();

    // XXX i don't think we're doing the leader calculation in a portable way
    //     we need a way to compare idx to the origin idx to figure out if this invocation represents the CTA leader
    agency::cuda::detail::inner_shared_parameter<InnerFactory> inner_param(idx[1] == 0, inner_factory_);

    // convert the references to raw references before passing them to f_
    f_(idx,
       static_cast<typename std::pointer_traits<DependencyPointers>::element_type&>(*agency::detail::get<Indices>(dependency_ptrs_))...,
       static_cast<typename std::pointer_traits<OuterArgumentPointer>::element_type&>(*outer_arg_ptr_),
       inner_param.get());
  }

  __device__
  void operator()()
  {
    impl(agency::detail::index_sequence_for<DependencyPointers...>());
  }
};

template<class Function, class IndexFunction, class OuterArgumentPointer, class InnerFactory, class... DependencyPointers>
__host__ __device__
when_all_execute_functor<Function, IndexFunction, OuterArgumentPointer, InnerFactory, DependencyPointers...>
  make_when_all_execute_functor(Function f, IndexFunction index_function, OuterArgumentPointer outer_arg_ptr, InnerFactory inner_factory, DependencyPointers... dependency_ptrs)
{
  return when_all_execute_functor<Function, IndexFunction, OuterArgumentPointer, InnerFactory, DependencyPointers...>(f, index_function, outer_arg_ptr, inner_factory, dependency_ptrs...);
}


struct call_data
{
  template<class Arg>
  __AGENCY_ANNOTATION
  auto operator()(Arg& arg) const
    -> decltype(arg.data())
  {
    return arg.data();
  }
};


template<class FutureOrFutureReference>
struct value_type_is_not_void
  : std::integral_constant<
      bool,
      !std::is_void<
        typename agency::future_traits<
          typename std::decay<FutureOrFutureReference>::type
        >::value_type
      >::value
    >
{};


template<class ResultPointer, class... Pointers>
struct move_construct_result_functor
{
  ResultPointer result_ptr_;
  agency::cuda::detail::tuple<Pointers...> ptrs_;

  using result_type = typename std::pointer_traits<ResultPointer>::element_type;

  __AGENCY_ANNOTATION
  move_construct_result_functor(ResultPointer result_ptr, Pointers... ptrs)
    : result_ptr_(result_ptr),
      ptrs_(ptrs...)
  {}

  template<size_t... Indices>
  __AGENCY_ANNOTATION
  inline void impl(agency::detail::index_sequence<Indices...>)
  {
    *result_ptr_ = result_type(std::move(*agency::detail::get<Indices>(ptrs_))...);
  }

  __AGENCY_ANNOTATION
  inline void operator()()
  {
    impl(agency::detail::index_sequence_for<Pointers...>());
  }
};

template<class ResultPointer, class... Pointers>
__AGENCY_ANNOTATION
move_construct_result_functor<ResultPointer,Pointers...>
  make_move_construct_result_functor(ResultPointer result_ptr, Pointers... ptrs)
{
  return move_construct_result_functor<ResultPointer,Pointers...>(result_ptr,ptrs...);
}


template<class Pointer, class... Pointers>
agency::cuda::future<
  agency::detail::when_all_result_t<
    agency::cuda::future<
      typename std::pointer_traits<Pointer>::element_type
    >,
    agency::cuda::future<
      typename std::pointer_traits<Pointers>::element_type
    >...
  >
>
  move_construct_result(agency::cuda::detail::event& dependency, Pointer ptr, Pointers... ptrs)
{
  using result_type = agency::detail::when_all_result_t<
    agency::cuda::future<typename std::pointer_traits<Pointer>::element_type>,
    agency::cuda::future<typename std::pointer_traits<Pointers>::element_type>...
  >;

  // create a state to hold the result
  agency::cuda::detail::asynchronous_state<result_type> result_state(agency::cuda::detail::construct_not_ready);

  // create a function to do the move construction
  auto f = make_move_construct_result_functor(result_state.data(), ptr, ptrs...);

  // launch the function
  agency::cuda::detail::stream stream;
  agency::cuda::detail::event result_event = dependency.then(f, dim3{1}, dim3{1}, 0, stream.native_handle());

  return agency::cuda::future<result_type>(std::move(stream), std::move(result_event), std::move(result_state));
}


inline agency::cuda::future<void>
  move_construct_result(agency::cuda::detail::event& dependency)
{
  return agency::cuda::future<void>(std::move(dependency), agency::cuda::detail::asynchronous_state<void>(agency::cuda::detail::construct_not_ready));
}


template<size_t... Indices, class TupleOfFutures>
auto move_construct_result_from_tuple_of_futures(agency::detail::index_sequence<Indices...>, agency::cuda::detail::event& dependency, TupleOfFutures& futures)
  -> decltype(move_construct_result(dependency, agency::detail::get<Indices>(futures).data()...))
{
  return move_construct_result(dependency, agency::detail::get<Indices>(futures).data()...);
}


template<class TypeList>
struct type_list_to_tuple_or_single_type_or_void;


template<class... Types>
struct type_list_to_tuple_or_single_type_or_void<agency::detail::type_list<Types...>>
{
  using type = agency::detail::tuple<Types...>;
};

template<class T>
struct type_list_to_tuple_or_single_type_or_void<agency::detail::type_list<T>>
{
  using type = T;
};

template<>
struct type_list_to_tuple_or_single_type_or_void<agency::detail::type_list<>>
{
  using type = void;
};


template<class IndexSequence, class TypeList>
struct new_when_all_execute_and_select_result_from_type_list;

template<size_t... Indices, class TypeList>
struct new_when_all_execute_and_select_result_from_type_list<agency::detail::index_sequence<Indices...>, TypeList>
{
  using unfiltered_type_list = agency::detail::type_list<
    agency::detail::type_list_element<Indices,TypeList>...
  >;

  template<class T>
  struct is_not_void : std::integral_constant<bool, !std::is_void<T>::value> {};

  using non_void_types = agency::detail::type_list_filter<is_not_void,unfiltered_type_list>;

  using type = typename type_list_to_tuple_or_single_type_or_void<non_void_types>::type;
};

template<class IndexSequence, class TupleOfFutures>
struct new_when_all_execute_and_select_result
{
  // get the tuple's type_list of futures
  using future_types = agency::detail::tuple_elements<TupleOfFutures>;

  template<class Future>
  struct future_value_type
  {
    using type = typename agency::future_traits<Future>::value_type;
  };

  // get each future's value_type
  using value_types = agency::detail::type_list_map<future_value_type, future_types>;

  using type = typename new_when_all_execute_and_select_result_from_type_list<IndexSequence, value_types>::type;
};

template<class IndexSequence, class TupleOfFutures>
using new_when_all_execute_and_select_result_t = typename new_when_all_execute_and_select_result<IndexSequence, TupleOfFutures>::type;


template<class TypeList>
struct new_when_all_result_from_type_list
{
  template<class T>
  struct is_not_void : std::integral_constant<bool, !std::is_void<T>::value> {};

  // filter out void
  using non_void_types = agency::detail::type_list_filter<is_not_void,TypeList>;

  using type = typename type_list_to_tuple_or_single_type_or_void<non_void_types>::type;
};


template<class TypeList>
using new_when_all_result_from_type_list_t = typename new_when_all_result_from_type_list<TypeList>::type;


template<class... Types>
struct new_when_all_result
{
  using type = new_when_all_result_from_type_list_t<agency::detail::type_list<Types...>>;
};


template<class... Types>
using new_when_all_result_t = typename new_when_all_result<Types...>::type;


template<class Function, class OuterArgPointer, class InnerFactory, class... Pointers>
agency::cuda::detail::event
  launch_when_all_execute_operation_impl(agency::cuda::detail::event& dependency, Function f, agency::cuda::grid_executor::shape_type shape, OuterArgPointer outer_arg_ptr, InnerFactory inner_factory, Pointers... ptrs)
{
  cudaStream_t stream = 0;

  // make a function implementing the operation
  auto continuation = make_when_all_execute_functor(f, agency::cuda::detail::this_index_1d{}, outer_arg_ptr, inner_factory, ptrs...);

  // convert the shape to CUDA types
  agency::uint3 outer_shape = agency::detail::shape_cast<agency::uint3>(agency::detail::get<0>(shape));
  agency::uint3 inner_shape = agency::detail::shape_cast<agency::uint3>(agency::detail::get<1>(shape));

  ::dim3 grid_dim{outer_shape[0], outer_shape[1], outer_shape[2]};
  ::dim3 block_dim{inner_shape[0], inner_shape[1], inner_shape[2]};

  // launch the continuation
  return dependency.then(continuation, grid_dim, block_dim, 0, stream);
}


template<size_t... Indices, class Function, class OuterArgumentPointer, class InnerFactory, class TupleOfNonVoidFutures>
agency::cuda::detail::event
  launch_when_all_execute_operation(agency::detail::index_sequence<Indices...>, agency::cuda::detail::event& dependency, Function f, agency::cuda::grid_executor::shape_type shape, OuterArgumentPointer outer_arg_ptr, InnerFactory inner_factory, TupleOfNonVoidFutures& futures)
{
  // unpack the futures and pass their data pointers to when_all_execute_impl3
  return launch_when_all_execute_operation_impl(dependency, f, shape, outer_arg_ptr, inner_factory, agency::cuda::detail::get<Indices>(futures).data()...);
}


template<size_t... SelectedIndices, size_t... TupleIndices, class TupleOfFutures, class Function, class OuterFactory, class InnerFactory>
agency::cuda::future<new_when_all_execute_and_select_result_t<agency::detail::index_sequence<SelectedIndices...>, TupleOfFutures>>
  when_all_execute_and_select_impl(agency::detail::index_sequence<SelectedIndices...>,
                                   agency::detail::index_sequence<TupleIndices...>,
                                   Function f,
                                   agency::cuda::grid_executor::shape_type shape,
                                   TupleOfFutures tuple_of_futures,
                                   OuterFactory outer_factory,
                                   InnerFactory inner_factory)
{
  // XXX we should static_assert that SelectedIndices are unique and in the correct range

  cudaStream_t stream = 0;

  // create a future to contain the outer argument
  using outer_arg_type = agency::detail::result_of_factory_t<OuterFactory>;
  auto outer_arg_future = agency::cuda::make_ready_future<outer_arg_type>(outer_factory());

  // join the events
  agency::cuda::detail::event when_all_ready = agency::cuda::detail::when_all(stream, outer_arg_future.event(), agency::detail::get<TupleIndices>(tuple_of_futures).event()...);

  // get a view of the non-void futures
  auto view_of_non_void_futures = agency::detail::tuple_filter_view<value_type_is_not_void>(tuple_of_futures);

  // launch the main operation
  auto when_all_execute_event = launch_when_all_execute_operation(agency::detail::make_tuple_indices(view_of_non_void_futures), when_all_ready, f, shape, outer_arg_future.data(), inner_factory, view_of_non_void_futures);

  // get a view of the selected futures
  auto view_of_selected_futures = agency::detail::forward_as_tuple(agency::detail::get<SelectedIndices>(tuple_of_futures)...);

  // get a view of the selected futures which are non-void
  auto view_of_selected_non_void_futures = agency::detail::tuple_filter_view<value_type_is_not_void>(view_of_selected_futures);

  // XXX we need to figure out how to safely end the futures' lifetimes
  //     we can't destroy them before their values have been moved into the result future
  //     we need to garbage collect them somehow
  //     or we need to take their data pointer and make the result construction kernel the owner

  // XXX to do this the right way, we'd move each future's pointer into the result construction kernel
  //     the agent that executes that operation would take ownership of the unique_ptrs and they would naturally get destroyed when that agent ended its computation
  //     we can't do that because that memory cannot be deallocated by a __device__ function
  // XXX we should implement a better memory allocator that can be used uniformly in __host__ & __device__ code

  return move_construct_result_from_tuple_of_futures(agency::detail::make_tuple_indices(view_of_selected_non_void_futures), when_all_execute_event, view_of_selected_non_void_futures);
}


template<class TupleOfFutures>
struct when_all_execute_result
{
  // turn TupleOfFutures into a type_list of the future types
  using future_types = agency::detail::tuple_elements<TupleOfFutures>;

  template<class Future>
  struct map_futures_to_value_type
  {
    using type = typename agency::future_traits<Future>::value_type;
  };

  // turn the type list of futures into a type list of their value_types
  using value_types = agency::detail::type_list_map<map_futures_to_value_type,future_types>;

  using type = new_when_all_result_from_type_list_t<value_types>;
};


template<class TupleOfFutures>
using when_all_execute_result_t = typename when_all_execute_result<TupleOfFutures>::type;


template<class Function, class TupleOfFutures, class OuterFactory, class InnerFactory>
agency::cuda::future<
  when_all_execute_result_t<
    typename std::decay<TupleOfFutures>::type
  >
>
  when_all_execute(Function f, agency::cuda::grid_executor::shape_type shape, TupleOfFutures&& futures, OuterFactory outer_factory, InnerFactory inner_factory)
{
  auto indices = agency::detail::make_tuple_indices(futures);
  return ::when_all_execute_and_select_impl(indices, indices, f, shape, std::move(futures), outer_factory, inner_factory);
}


int main()
{
  auto factory = [] __device__ { return 7; };

  {
    // int, float -> (int, float)
    agency::cuda::future<int>   f1 = agency::cuda::make_ready_future<int>(7);
    agency::cuda::future<float> f2 = agency::cuda::make_ready_future<float>(13);

    auto f3 = when_all_execute([] __device__ (agency::cuda::grid_executor::index_type idx, int& past_arg1, float& past_arg2, int& outer_arg, int& inner_arg)
    {
      if(idx[0] == 0 && idx[1] == 0)
      {
        ++past_arg1;
        ++past_arg2;
      }
    },
    agency::cuda::grid_executor::shape_type{100,256},
    agency::detail::make_tuple(std::move(f1),std::move(f2)),
    factory,
    factory);

    auto result = f3.get();

    assert(result == agency::detail::make_tuple(8,14));
  }

  {
    // int, void -> int
    agency::cuda::future<int>  f1 = agency::cuda::make_ready_future<int>(7);
    agency::cuda::future<void> f2 = agency::cuda::make_ready_future();

    auto f3 = when_all_execute([] __device__ (agency::cuda::grid_executor::index_type idx, int& past_arg, int& outer_arg, int& inner_arg)
    {
      if(idx[0] == 0 && idx[1] == 0)
      {
        ++past_arg;
      }
    },
    agency::cuda::grid_executor::shape_type{100,256},
    agency::detail::make_tuple(std::move(f1),std::move(f2)),
    factory,
    factory);

    auto result = f3.get();

    assert(result == 8);
  }

  {
    // void, int -> int
    agency::cuda::future<void> f1 = agency::cuda::make_ready_future();
    agency::cuda::future<int>  f2 = agency::cuda::make_ready_future<int>(7);

    auto f3 = when_all_execute([] __device__ (agency::cuda::grid_executor::index_type idx, int& past_arg, int& outer_arg, int& inner_arg)
    {
      if(idx[0] == 0 && idx[1] == 0)
      {
        ++past_arg;
      }
    },
    agency::cuda::grid_executor::shape_type{100,256},
    agency::detail::make_tuple(std::move(f1),std::move(f2)),
    factory,
    factory);

    auto result = f3.get();

    assert(result == 8);
  }

  {
    // void -> void
    agency::cuda::future<void> f1 = agency::cuda::make_ready_future();

    auto f3 = when_all_execute([] __device__ (agency::cuda::grid_executor::index_type idx, int& outer_arg, int& inner_arg)
    {
    },
    agency::cuda::grid_executor::shape_type{100,256},
    agency::detail::make_tuple(std::move(f1)),
    factory,
    factory);

    f3.get();
  }

  {
    // void, void -> void
    agency::cuda::future<void> f1 = agency::cuda::make_ready_future();
    agency::cuda::future<void> f2 = agency::cuda::make_ready_future();

    auto f3 = when_all_execute([] __device__ (agency::cuda::grid_executor::index_type idx, int& outer_arg, int& inner_arg)
    {
    },
    agency::cuda::grid_executor::shape_type{100,256},
    agency::detail::make_tuple(std::move(f1), std::move(f2)),
    factory,
    factory);

    f3.get();
  }

  {
    // -> void

    auto f3 = when_all_execute([] __device__ (agency::cuda::grid_executor::index_type idx, int& outer_arg, int& inner_arg)
    {
    },
    agency::cuda::grid_executor::shape_type{100,256},
    agency::detail::make_tuple(),
    factory,
    factory);

    f3.get();
  }

  std::cout << "OK" << std::endl;

  return 0;
}

