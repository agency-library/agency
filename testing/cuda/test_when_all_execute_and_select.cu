#include <agency/cuda/grid_executor.hpp>
#include <agency/cuda/detail/when_all_execute_and_select.hpp>
#include <memory>


template<class TypeList>
struct when_all_result_from_type_list
{
  template<class T>
  using is_not_void = std::integral_constant<bool, !std::is_void<T>::value>;

  // filter void types
  using filtered_type_list = agency::detail::type_list_filter<is_not_void,TypeList>;

  // get a tuple, or single type of the filtered types
  using type = agency::detail::tuple_or_single_type_or_void_from_type_list_t<filtered_type_list>;
};

template<class TypeList>
using when_all_result_from_type_list_t = typename when_all_result_from_type_list<TypeList>::type;


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

  using type = when_all_result_from_type_list_t<value_types>;
};


template<class TupleOfFutures>
using when_all_execute_result_t = typename when_all_execute_result<TupleOfFutures>::type;


template<size_t... Indices, class Function, class TupleOfFutures, class OuterFactory, class InnerFactory>
agency::cuda::future<
  when_all_execute_result_t<
    typename std::decay<TupleOfFutures>::type
  >
>
  when_all_execute_impl(agency::detail::index_sequence<Indices...> indices, Function f, agency::cuda::grid_executor::shape_type shape, TupleOfFutures&& futures, OuterFactory outer_factory, InnerFactory inner_factory)
{
  agency::cuda::grid_executor exec;
  return exec.template when_all_execute_and_select<Indices...>(f, shape, std::forward<TupleOfFutures>(futures), outer_factory, inner_factory);
}


template<class Function, class TupleOfFutures, class OuterFactory, class InnerFactory>
agency::cuda::future<
  when_all_execute_result_t<
    typename std::decay<TupleOfFutures>::type
  >
>
  when_all_execute(Function f, agency::grid_executor::shape_type shape, TupleOfFutures&& futures, OuterFactory outer_factory, InnerFactory inner_factory)
{
  auto indices = agency::detail::make_tuple_indices(futures);
  return ::when_all_execute_impl(indices, f, shape, index_function, std::forward<TupleOfFutures>(futures), outer_factory, inner_factory);
}


int main()
{
  auto factory = [] __device__ { return 7; };
  auto shape = agency::cuda::grid_executor::shape_type{100,256};

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
    shape,
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
    shape,
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
    shape,
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
    shape,
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
    shape,
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
    shape,
    agency::detail::make_tuple(),
    factory,
    factory);

    f3.get();
  }

  std::cout << "OK" << std::endl;

  return 0;
}

