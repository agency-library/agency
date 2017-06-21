#pragma once

#include <agency/detail/config.hpp>
#include <agency/experimental/optional.hpp>
#include <agency/container/bulk_result.hpp>
#include <agency/detail/index.hpp>
#include <agency/detail/shape.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <utility>
#include <tuple>
#include <type_traits>

namespace agency
{
namespace detail
{


constexpr size_t invalid_scope{100000};


} // end detail


template<size_t scope, class T>
class scope_result;


template<class T>
class scope_result<detail::invalid_scope,T> {};


namespace detail
{


template<class T>
using no_result_t = scope_result<detail::invalid_scope,T>;


} // end detail


template<class T>
__AGENCY_ANNOTATION
detail::no_result_t<T> no_result();


template<size_t N, class T>
class scope_result : public experimental::optional<T>
{
  private:
    using super_t = experimental::optional<T>;

  public:
    using result_type = T;

    static constexpr size_t scope = N;

    __AGENCY_ANNOTATION
    scope_result(scope_result&& other)
      : super_t(std::move(other))
    {}

    __AGENCY_ANNOTATION
    scope_result(const T& result)
      : super_t(result)
    {}

    __AGENCY_ANNOTATION
    scope_result(T&& result)
      : super_t(std::move(result))
    {}

    __AGENCY_ANNOTATION
    scope_result(const decltype(std::ignore)&)
      : scope_result()
    {}

    __AGENCY_ANNOTATION
    scope_result(const detail::no_result_t<T>&)
      : scope_result()
    {}

  private:
    __AGENCY_ANNOTATION
    scope_result()
      : super_t(experimental::nullopt)
    {}

    template<class U>
    __AGENCY_ANNOTATION
    friend detail::no_result_t<U> no_result();
};


template<class T>
__AGENCY_ANNOTATION
detail::no_result_t<T> no_result()
{
  return detail::no_result_t<T>();
}


namespace detail
{


template<class T>
struct is_scope_result : std::false_type {};

template<size_t scope,class T>
struct is_scope_result<scope_result<scope,T>> : std::true_type {};


template<size_t scope, class T, class Executor>
class scope_result_container
  : public bulk_result<
      T,
      shape_take_t<scope, executor_shape_t<Executor>>,
      executor_allocator_t<Executor, T>
    >
{
  private:
    using super_t = bulk_result<
      T,
      shape_take_t<scope, executor_shape_t<Executor>>,
      executor_allocator_t<Executor, T>
    >;

    using base_index_type = typename super_t::index_type;

  public:
    static_assert(scope < executor_execution_depth<Executor>::value, "scope_result_container: scope must be less than Executor's execution_depth");

    using result_type = super_t;

    using shape_type = executor_shape_t<Executor>;
    using index_type = executor_index_t<Executor>;

    __agency_exec_check_disable__
    scope_result_container()
      : super_t{}
    {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    scope_result_container(const scope_result_container& other)
      : super_t(other)
    {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    scope_result_container(scope_result_container&& other)
      : super_t(std::move(other))
    {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    scope_result_container(const shape_type& shape)
      : super_t{detail::shape_take<scope>(shape)}
    {}

    struct reference
    {
      super_t& self;
      base_index_type idx;

      __agency_exec_check_disable__
      __AGENCY_ANNOTATION
      void operator=(scope_result<scope,T>&& result)
      {
        if(result)
        {
          self[idx] = std::move(*result);
        }
      }
    };

    __AGENCY_ANNOTATION
    reference operator[](const index_type& idx)
    {
      // take the first scope elements of the incoming index
      // to produce an index into the underlying array
      auto base_idx = detail::index_take<scope>(idx);
      return reference{*this,base_idx};
    }
};


// special case for scope 0: there is only a single result
template<class T, class Executor>
class scope_result_container<0, T, Executor>
{
  private:
    // XXX not making this a base class might make it difficult to cast
    //     future<scope_result_container<T,...>> to future<T>
    T single_element_;

  public:
    using shape_type = executor_shape_t<Executor>;
    using index_type = executor_index_t<Executor>;

    using result_type = T;

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    scope_result_container()
      : single_element_{}
    {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    scope_result_container(const scope_result_container& other)
      : single_element_(other.single_element_)
    {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    scope_result_container(const shape_type&)
      : scope_result_container()
    {}

    __AGENCY_ANNOTATION
    scope_result_container& operator[](const index_type&)
    {
      return *this;
    }

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    void operator=(scope_result<0,T>&& result)
    {
      if(result)
      {
        single_element_ = std::move(*result);
      }
    }

    __AGENCY_ANNOTATION
    operator result_type&& () &&
    {
      return std::move(single_element_);
    }
};


// this maps a scope_result<T,N> returned by a user function
// to the intermediate scope_result_container type used between the execution policy
// and executor. it is not the type returned by bulk_invoke
template<class ScopeResult, class Executor, bool Enable = is_scope_result<ScopeResult>::value>
struct scope_result_to_scope_result_container
{
  template<class T>
  using member_result_type = typename T::result_type;

  // the type returned by the user function
  using user_result_type = member_result_type<ScopeResult>;

  // the scope at which the result is returned
  static constexpr size_t scope = ScopeResult::scope;

  // the type returned by bulk_invoke()
  using type = scope_result_container<
    scope,
    user_result_type,
    Executor
  >;
};


// when T isn't a scope_result, it just returns some dummy type
template<class ScopeResult, class Executor>
struct scope_result_to_scope_result_container<ScopeResult,Executor,false>
{
  struct dummy_container
  {
    struct result_type {};
  };

  using type = dummy_container;
};


// this maps a scope_result<T,N> returned by a user function
// to the type of result returned by bulk_invoke()
template<class ScopeResult, class Executor>
struct scope_result_to_bulk_invoke_result
{
  using scope_result_container = typename scope_result_to_scope_result_container<ScopeResult,Executor>::type;

  using type = typename scope_result_container::result_type;
};


} // end detail
} // end agency

