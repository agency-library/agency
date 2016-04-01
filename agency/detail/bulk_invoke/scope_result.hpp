#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/optional.hpp>
#include <agency/detail/array.hpp>
#include <agency/detail/index.hpp>
#include <agency/detail/shape.hpp>
#include <utility>
#include <tuple>
#include <type_traits>

namespace agency
{
namespace detail
{


constexpr size_t invalid_scope{100000};


} // end detail


template<class T, size_t scope>
class scope_result;


template<class T>
class scope_result<T,detail::invalid_scope> {};


template<class T>
__AGENCY_ANNOTATION
scope_result<T,detail::invalid_scope> no_result();


template<class T, size_t N>
class scope_result : public detail::optional<T>
{
  private:
    using super_t = detail::optional<T>;

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
    scope_result(const scope_result<T,detail::invalid_scope>&)
      : scope_result()
    {}

  private:
    __AGENCY_ANNOTATION
    scope_result()
      : super_t(detail::nullopt)
    {}

    template<class U>
    __AGENCY_ANNOTATION
    friend scope_result<U,detail::invalid_scope> no_result();
};


template<class T>
__AGENCY_ANNOTATION
scope_result<T,detail::invalid_scope> no_result()
{
  return scope_result<T,detail::invalid_scope>();
}


namespace detail
{


template<class T>
struct is_scope_result : std::false_type {};

template<class T, size_t scope>
struct is_scope_result<scope_result<T,scope>> : std::true_type {};


template<class T, size_t scope, class Executor>
class scope_result_container
  : public detail::array<
      T,
      shape_take_t<scope, executor_shape_t<Executor>>,
      typename executor_traits<Executor>::template allocator<T>,
      index_take_t<scope, executor_index_t<Executor>>
    >
{
  private:
    using super_t = detail::array<
      T,
      shape_take_t<scope, executor_shape_t<Executor>>,
      typename executor_traits<Executor>::template allocator<T>,
      index_take_t<scope, executor_index_t<Executor>>
    >;

    using base_index_type = typename super_t::index_type;

  public:
    static_assert(scope < executor_traits<Executor>::execution_depth, "scope_result_container: scope must be less than Executor's execution_depth");

    using result_type = super_t;

    using shape_type = typename executor_traits<Executor>::shape_type;
    using index_type = typename executor_traits<Executor>::index_type;

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
    scope_result_container(const shape_type& shape)
      : super_t{detail::shape_take<scope>(shape)}
    {}

    struct reference
    {
      super_t& self;
      base_index_type idx;

      __agency_exec_check_disable__
      __AGENCY_ANNOTATION
      void operator=(scope_result<T,scope>&& result)
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
class scope_result_container<T, 0, Executor>
{
  private:
    // XXX not making this a base class might make it difficult to cast
    //     future<scope_result_container<T,...>> to future<T>
    T single_element_;

  public:
    using shape_type = typename executor_traits<Executor>::shape_type;
    using index_type = typename executor_traits<Executor>::index_type;

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
    scope_result_container& operator[](const index_type& idx)
    {
      return *this;
    }

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    void operator=(scope_result<T,0>&& result)
    {
      if(result)
      {
        single_element_ = std::move(*result);
      }
    }

    __AGENCY_ANNOTATION
    operator result_type& () &
    {
      return single_element_;
    }

    __AGENCY_ANNOTATION
    operator const result_type& () const &
    {
      return single_element_;
    }

    // XXX this might be the only conversion we actually want
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
  using nested_result_type = typename T::result_type;

  // the type returned by the user function
  using user_result_type = nested_result_type<ScopeResult>;

  // the scope at which the result is returned
  static constexpr size_t scope = ScopeResult::scope;

  // the type returned by bulk_invoke()
  using type = scope_result_container<
    user_result_type,
    scope,
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

