#include <type_traits>
#include <typeinfo>
#include <iostream>


template<int i, typename T1, typename... Types>
  struct __initializer : __initializer<i+1,Types...>
{
  typedef __initializer<i+1,Types...> super_t;

  using super_t::operator();

  int operator()(void *ptr, const T1& other)
  {
    new (ptr) T1(other);
    return i;
  }
};


template<int i, typename T>
struct __initializer<i,T>
{
  int operator()(void* ptr, const T& other)
  {
    new (ptr) T(other);
    return i;
  }
};


template<int i, int... js>
struct __constexpr_max
{
  static const int value = i < __constexpr_max<js...>::value ? __constexpr_max<js...>::value : i;
};


template<int i>
struct __constexpr_max<i>
{
  static const int value = i;
};


template<typename T1, typename... Types>
class variant;


template<int i, typename Variant> struct variant_element;


template<int i, typename T0, typename... Types>
struct variant_element<i, variant<T0, Types...>>
  : variant_element<i-1,Types...>
{};


template<typename T0, typename... Types>
struct variant_element<0, variant<T0, Types...>>
{
  typedef T0 type;
};


template<int i, typename... Types>
using variant_element_t = typename variant_element<i,Types...>::type;


template<typename T, typename U>
struct __propagate_reference;


template<typename T, typename U>
struct __propagate_reference<T&, U>
{
  typedef U& type;
};


template<typename T, typename U>
struct __propagate_reference<const T&, U>
{
  typedef const U& type;
};


template<typename T, typename U>
struct __propagate_reference<T&&, U>
{
  typedef U&& type;
};


template<int i, typename VariantReference>
struct __variant_element_reference
  : __propagate_reference<
      VariantReference,
      variant_element_t<
        i,
        typename std::decay<VariantReference>::type
      >
    >
{};


template<int i, typename VariantReference>
using __variant_element_reference_t = typename __variant_element_reference<i,VariantReference>::type;


template<typename Visitor, typename Variant>
auto apply_visitor(Visitor visitor, Variant&& var) ->
  typename std::result_of<
    Visitor(__variant_element_reference_t<0,decltype(var)>)
  >::type;


template<typename Visitor, typename Variant1, typename Variant2>
auto apply_visitor(Visitor visitor, Variant1&& var1, Variant2&& var2) ->
  typename std::result_of<
    Visitor(__variant_element_reference_t<0,decltype(var1)>,
            __variant_element_reference_t<0,decltype(var2)>)
  >::type;


template<int i, typename T, typename... Types>
struct __index_of_impl;


// no match, keep going
template<int i, typename T, typename U, typename... Types>
struct __index_of_impl<i,T,U,Types...>
  : __index_of_impl<i+1,T,Types...>
{};


// found a match
template<int i, typename T, typename... Types>
struct __index_of_impl<i,T,T,Types...>
{
  static const int value = i;
};


// no match
template<int i, typename T>
struct __index_of_impl<i,T>
{
  static const int value = i;
};


template<typename T, typename... Types>
using __index_of = __index_of_impl<0,T,Types...>;


template<typename T1, typename... Types>
class variant
{
  private:
    struct destroyer
    {
      template<typename T>
      typename std::enable_if<
        !std::is_pod<T>::value
      >::type
        operator()(T& x)
      {
        x.~T();
      }
    
      template<typename T>
      typename std::enable_if<
        std::is_pod<T>::value
      >::type
        operator()(T& x)
      {
        // do nothing for POD types
      }
    };

  public:
    variant() : variant(T1{}) {}

  private:
    struct placement_mover
    {
      void *ptr;

      placement_mover(void* p) : ptr(p) {}

      template<typename U>
      int operator()(U&& x)
      {
        // decay off the reference of U, if any
        using T = typename std::decay<U>::type;
        new (ptr) T(std::move(x));
        return __index_of<T,T1,Types...>::value;
      }
    };

  public:
    variant(variant&& other)
    {
      which_ = apply_visitor(placement_mover(data()), std::move(other));
    }

  private:
    struct placement_copier
    {
      void *ptr;

      placement_copier(void* p) : ptr(p) {}

      template<typename U>
      int operator()(U&& x)
      {
        // decay off the reference of U, if any
        using T = typename std::decay<U>::type;
        new (ptr) T(x);
        return __index_of<T,T1,Types...>::value;
      }
    };

  public:
    variant(const variant& other)
      : which_(apply_visitor(placement_copier(data()), other))
    {}

    template<typename... Args>
    variant(Args&&... args)
    {
      __initializer<0, T1, Types...> init;
      which_ = init(data(), std::forward<Args>(args)...);
    }

    ~variant()
    {
      apply_visitor(destroyer(), *this);
    }

    int which() const
    {
      return which_;
    }

  private:
    struct return_type_info
    {
      template<typename T>
      const std::type_info& operator()(const T& x)
      {
        return typeid(x);
      }
    };

  public:
    const std::type_info& type() const
    {
      return apply_visitor(return_type_info(), *this);
    }

  private:
    struct equals
    {
      template<typename U1, typename U2>
      bool operator()(const U1&, const U2&)
      {
        return false;
      }

      template<typename T>
      bool operator()(const T& lhs, const T& rhs)
      {
        return lhs == rhs;
      }
    };


  public:
    bool operator==(const variant& rhs) const
    {
      return which() == rhs.which() && apply_visitor(equals(), *this, rhs);
    }

    bool operator!=(const variant& rhs) const
    {
      return !operator==(rhs);
    }

  private:
    struct less
    {
      template<typename U1, typename U2>
      bool operator()(const U1&, const U2&)
      {
        return false;
      }

      template<typename T>
      bool operator()(const T& lhs, const T& rhs)
      {
        return lhs < rhs;
      }
    };

  public:
    bool operator<(const variant& rhs) const
    {
      if(which() != rhs.which()) return which() < rhs.which();

      return apply_visitor(less(), *this, rhs);
    }

    bool operator<=(const variant& rhs) const
    {
      return !(rhs < *this);
    }

    bool operator>(const variant& rhs) const
    {
      return rhs < *this;
    }

    bool operator>=(const variant& rhs) const
    {
      return !(*this < rhs);
    }

  private:
    typedef typename std::aligned_storage<
      __constexpr_max<sizeof(T1), sizeof(Types)...>::value
    >::type storage_type;

    storage_type storage_;

    void *data()
    {
      return &storage_;
    }

    const void *data() const
    {
      return &storage_;
    }

    int which_;
};


struct __ostream_output_visitor
{
  std::ostream &os;

  __ostream_output_visitor(std::ostream& os) : os(os) {}

  template<typename T>
  std::ostream& operator()(const T& x)
  {
    return os << x;
  }
};


template<typename T1, typename... Types>
std::ostream &operator<<(std::ostream& os, const variant<T1,Types...>& v)
{
  return apply_visitor(__ostream_output_visitor(os), v);
}


template<typename Visitor, typename Result, typename T, typename... Types>
struct __apply_visitor_impl : __apply_visitor_impl<Visitor,Result,Types...>
{
  typedef __apply_visitor_impl<Visitor,Result,Types...> super_t;

  static Result do_it(Visitor visitor, void* ptr, int which)
  {
    if(which == 0)
    {
      return visitor(*reinterpret_cast<T*>(ptr));
    }

    return super_t::do_it(visitor, ptr, --which);
  }


  static Result do_it(Visitor visitor, const void* ptr, int which)
  {
    if(which == 0)
    {
      return visitor(*reinterpret_cast<const T*>(ptr));
    }

    return super_t::do_it(visitor, ptr, --which);
  }
};


template<typename Visitor, typename Result, typename T>
struct __apply_visitor_impl<Visitor,Result,T>
{
  static Result do_it(Visitor visitor, void* ptr, int)
  {
    return visitor(*reinterpret_cast<T*>(ptr));
  }

  static Result do_it(Visitor visitor, const void* ptr, int)
  {
    return visitor(*reinterpret_cast<const T*>(ptr));
  }
};


template<typename Visitor, typename Result, typename Variant>
struct __apply_visitor;


template<typename Visitor, typename Result, typename T1, typename... Types>
struct __apply_visitor<Visitor,Result,variant<T1,Types...>>
  : __apply_visitor_impl<Visitor,Result,T1,Types...>
{};


template<typename Visitor, typename Variant>
auto apply_visitor(Visitor visitor, Variant&& var) ->
  typename std::result_of<
    Visitor(__variant_element_reference_t<0,decltype(var)>)
  >::type
{
  using result_type = typename std::result_of<
    Visitor(__variant_element_reference_t<0,decltype(var)>)
  >::type;
 
  using impl = __apply_visitor<Visitor,result_type,typename std::decay<Variant>::type>;

  return impl::do_it(visitor, &var, var.which());
}



template<typename Visitor, typename Result, typename ElementReference>
struct __unary_visitor_binder
{
  Visitor visitor;
  ElementReference x;

  __unary_visitor_binder(Visitor visitor, ElementReference x) : visitor(visitor), x(x) {}

  template<typename T>
  Result operator()(T&& y)
  {
    return visitor(x, std::forward<T>(y));
  }
};


template<typename Visitor, typename Result, typename VariantReference>
struct __binary_visitor_binder
{
  Visitor visitor;
  VariantReference y;

  __binary_visitor_binder(Visitor visitor, VariantReference ref) : visitor(visitor), y(ref) {}

  template<typename T>
  Result operator()(T&& x)
  {
    return apply_visitor(__unary_visitor_binder<Visitor, Result, decltype(x)>(visitor, std::forward<T>(x)), y);
  }
};


template<typename Visitor, typename Variant1, typename Variant2>
auto apply_visitor(Visitor visitor, Variant1&& var1, Variant2&& var2) ->
  typename std::result_of<
    Visitor(__variant_element_reference_t<0,decltype(var1)>,
            __variant_element_reference_t<0,decltype(var2)>)
  >::type
{
  using result_type = typename std::result_of<
    Visitor(__variant_element_reference_t<0,decltype(var1)>,
            __variant_element_reference_t<0,decltype(var2)>)
  >::type;

  auto visitor_wrapper = __binary_visitor_binder<Visitor,result_type,decltype(var2)>(visitor, std::forward<Variant2>(var2));

  return apply_visitor(visitor_wrapper, std::forward<Variant1>(var1));
}

