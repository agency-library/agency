#include <agency/sequential_executor.hpp>
#include <agency/executor_traits.hpp>
#include <agency/detail/tuple.hpp>
#include <iostream>
#include <cassert>
#include <functional>

template<class T, class... Args>
struct shared_parameter
{
  T make() const
  {
    return __tu::make_from_tuple<T>(args_);
  }

  agency::detail::tuple<Args...> args_;
};

template<class T> struct is_shared_parameter : std::false_type {};
template<class T, class... Args>
struct is_shared_parameter<shared_parameter<T,Args...>> : std::true_type {};

template<class T>
struct is_shared_parameter_ref
  : std::integral_constant<
      bool,
      (std::is_reference<T>::value && is_shared_parameter<typename std::remove_reference<T>::type>::value)
    >
{};

template<class T, class... Args>
shared_parameter<T,Args...> share(Args&&... args)
{
  return shared_parameter<T,Args...>{std::make_tuple(std::forward<Args>(args)...)};
}

template<class T>
shared_parameter<T,T> share(const T& val)
{
  return shared_parameter<T,T>{std::make_tuple(val)};
}


//template<size_t... I, class Function, class... SharedArgs>
//void bulk_invoke(agency::detail::index_sequence<I...>, Function f, size_t n, SharedArgs&&... shared_args)
//{
//  auto exec = agency::sequential_executor{};
//
//  // explicitly construct the shared parameter
//  // it gets copy constructed by the executor
//  // XXX problems with this approach
//  //     1. the type of the shared parameter is constructed twice
//  //       1.1 we can ameliorate this if executors receive shared parameters as forwarding references
//  //           and move them into exec.bulk_invoke()
//  //     2. requires the type of the shared parameter to be copy constructable
//  //       2.1 we can fix this if executors receive shared parameters as forwarding references
//  //     3. won't be able to support concurrent construction
//
//  // make the shared variable initializers and tuple them up
//  auto shared_inits = agency::detail::make_tuple(std::forward<SharedArgs>(shared_args).make()...);
//  
//  exec.bulk_invoke([=](size_t idx, decltype(shared_inits)& shared_arg_tuple)
//  {
//    f(idx, std::get<I>(shared_arg_tuple)...);
//  },
//  n,
//  shared_inits);
//}


struct call_make
{
  template<class T, class... Args>
  T operator()(const shared_parameter<T,Args...>& parm) const
  {
    return parm.make();
  }
};


template<class Function, class SharedArgTuple>
void bulk_invoke_impl(Function f, size_t n, SharedArgTuple&& shared_arg_tuple)
{
  using executor_type = agency::sequential_executor;
  executor_type exec{};

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
  auto shared_init = __tu::tuple_map(call_make{}, std::forward<SharedArgTuple>(shared_arg_tuple));

  using shared_param_type = typename agency::executor_traits<executor_type>::template shared_param_type<decltype(shared_init)>;
  
  exec.bulk_invoke([=](size_t idx, shared_param_type& shared_params)
  {
    f(idx, shared_params);
  },
  n,
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


template<size_t... I, class Function, class... Args>
auto bind_unshared_parameters_impl(agency::detail::index_sequence<I...>, Function f, Args&&... args)
  -> decltype(
       std::bind(f, hold_shared_parameters_place<I>(std::forward<Args>(args))...)
     )
{
  return std::bind(f, hold_shared_parameters_place<I>(std::forward<Args>(args))...);
}


template<class Function, class... Args>
auto bind_unshared_parameters(Function f, Args&&... args)
  -> decltype(
       bind_unshared_parameters_impl(agency::detail::index_sequence_for<Args...>{}, f, std::forward<Args>(args)...)
     )
{
  return bind_unshared_parameters_impl(agency::detail::index_sequence_for<Args...>{}, f, std::forward<Args>(args)...);
}


template<class Function, class... Args>
void bulk_invoke(Function f, size_t n, Args&&... args)
{
  // the _1 is for the idx parameter
  auto g = bind_unshared_parameters(f, std::placeholders::_1, std::forward<Args>(args)...);

  // make a tuple of the shared args
  auto shared_arg_tuple = forward_shared_parameters_as_tuple(std::forward<Args>(args)...);

  // create h which takes a tuple of args and calls g
  auto h = unpack_shared_args_and_invoke<decltype(g)>{g};

  bulk_invoke_impl(h, n, shared_arg_tuple);
}


int main()
{
  auto lambda = [](int idx, int& shared0, int& shared1)
  {
    std::cout << "idx: " << idx << std::endl;
    std::cout << "shared0: " << shared0 << std::endl;
    std::cout << "shared1: " << shared1 << std::endl;

    assert(idx == shared0);
    assert(idx == -shared1);

    ++shared0;
    --shared1;
  };

  bulk_invoke(lambda, 10, share(0), share(0));

  return 0;
}

