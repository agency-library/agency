#include <agency/executor_traits.hpp>
#include <agency/execution_agent.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/index_cast.hpp>
#include <agency/detail/shape_cast.hpp>
#include <agency/detail/shared_parameter.hpp>
#include <agency/sequential_executor.hpp>
#include <agency/execution_policy.hpp>
#include <agency/detail/is_call_possible.hpp>
#include <iostream>
#include <cassert>
#include <functional>


template<size_t level, class T, class... Args>
agency::detail::shared_parameter<level, T,Args...> share(Args&&... args)
{
  return agency::detail::shared_parameter<level, T,Args...>{agency::detail::make_tuple(std::forward<Args>(args)...)};
}

template<size_t level, class T>
agency::detail::shared_parameter<level,T,T> share(const T& val)
{
  return agency::detail::shared_parameter<level,T,T>{agency::detail::make_tuple(val)};
}


template<class Function>
struct unpack_shared_parameters_from_executor_and_invoke
{
  mutable Function g;

  template<class Index, class... Types>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Types&... packaged_shared_params) const
  {
    auto shared_params = agency::detail::unpack_shared_parameters_from_executor(packaged_shared_params...);

    // XXX the following is the moral equivalent of:
    // g(idx, shared_params...);

    // create one big tuple of the arguments so we can just call tuple_apply
    auto idx_and_shared_params = __tu::tuple_prepend_invoke(shared_params, idx, agency::detail::forwarder{});

    __tu::tuple_apply(g, idx_and_shared_params);
  }
};


template<class Executor, class Function, class Tuple, size_t... TupleIndices>
void bulk_invoke_executor_impl(Executor& exec, Function f, typename agency::executor_traits<Executor>::shape_type shape, Tuple&& shared_init_tuple, agency::detail::index_sequence<TupleIndices...>)
{
  return agency::executor_traits<Executor>::execute(exec, unpack_shared_parameters_from_executor_and_invoke<Function>{f}, shape, agency::detail::get<TupleIndices>(std::forward<Tuple>(shared_init_tuple))...);
}


template<class Executor, class Function, class... Args>
typename agency::detail::enable_if_bulk_invoke_executor<
  Executor, Function, Args...
>::type
  bulk_invoke_executor(Executor& exec, Function f, typename agency::executor_traits<Executor>::shape_type shape, Args&&... args)
{
  // the _1 is for the executor idx parameter, which is the first parameter passed to f
  auto g = bind_unshared_parameters(f, std::placeholders::_1, std::forward<Args>(args)...);

  // make a tuple of the shared args
  auto shared_arg_tuple = forward_shared_parameters_as_tuple(std::forward<Args>(args)...);

  using traits = agency::executor_traits<Executor>;

  // package up the shared parameters for the executor
  const size_t executor_depth = agency::detail::execution_depth<
    typename traits::execution_category
  >::value;

  // construct shared arguments and package them for the executor
  auto factory_tuple = agency::detail::make_shared_parameter_factory_tuple<executor_depth>(shared_arg_tuple);

  ::bulk_invoke_executor_impl(exec, g, shape, factory_tuple, agency::detail::make_index_sequence<executor_depth>());
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

  ::bulk_invoke_executor(exec, lambda, shape, share<0>(1), share<1>(2), share<2>(3));
}


//void test2()
//{
//  auto lambda = [](agency::sequential_group<agency::sequential_agent>& self, int& outer_shared, int& inner_shared)
//  {
//    std::cout << "idx: " << self.index() << std::endl;
//    std::cout << "outer_shared: " << outer_shared << std::endl;
//    std::cout << "inner_shared: " << inner_shared << std::endl;
//  };
//
//  auto policy = agency::seq(2, agency::seq(2));
//
//  agency::bulk_invoke(policy, lambda, agency::share<0>(1), agency::share<1>(2));
//}


void test0()
{
  auto lambda = [](int idx, int& shared)
  {
    std::cout << "idx: " << idx << std::endl;

    assert(shared == idx + 13);
    ++shared;
  };

  agency::sequential_executor exec;
  ::bulk_invoke_executor(exec, lambda, 10, agency::share<0>(13));
}


int main()
{
//  test0();
  test1();
//  test2();

  return 0;
}

