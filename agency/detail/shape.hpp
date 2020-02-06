#pragma once

#include <agency/detail/config.hpp>
#include <agency/tuple.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/type_list.hpp>
#include <agency/detail/shape_tuple.hpp>
#include <agency/detail/utility.hpp>
#include <agency/container/array.hpp>

// we can't use std::numeric_limits<T>::max() in a __device__
// function, so we need to use an alternative in Thrust
#ifdef __NVCC__
#include <thrust/detail/integer_traits.h>
#else
#include <limits>
#endif

#include <tuple>

namespace agency
{
namespace detail
{


template<class Integral>
__AGENCY_ANNOTATION
typename std::enable_if<
  !detail::is_tuple_like<Integral>::value,
  Integral
>::type
  max_shape_dimensions()
{
#ifdef __NVCC__
  // XXX should avoid using thrust
  return thrust::detail::integer_traits<Integral>::const_max;
#else
  return std::numeric_limits<Integral>::max();
#endif
} // end max_shape_dimensions()


template<class ShapeTuple>
__AGENCY_ANNOTATION
typename std::enable_if<
  detail::is_tuple_like<ShapeTuple>::value,
  ShapeTuple
>::type
  max_shape_dimensions();


struct max_shape_dimensions_functor
{
  template<class T>
  __AGENCY_ANNOTATION
  T operator()(const T&) const
  {
    return detail::max_shape_dimensions<T>();
  }
}; // end max_shape_dimensions_functor


template<class ShapeTuple>
__AGENCY_ANNOTATION
typename std::enable_if<
  detail::is_tuple_like<ShapeTuple>::value,
  ShapeTuple
>::type
  max_shape_dimensions()
{
  return __tu::tuple_map_with_make(max_shape_dimensions_functor(), maker<ShapeTuple>(), ShapeTuple());
} // end max_shape_dimensions()


template<class Integral>
__AGENCY_ANNOTATION
typename std::enable_if<
  !detail::is_tuple_like<Integral>::value,
  Integral
>::type
  max_sizes(const Integral& max_size)
{
  return max_size;
} // end max_sizes()


struct max_sizes_functor
{
  struct max_functor
  {
    __AGENCY_ANNOTATION
    size_t operator()(size_t init, size_t x)
    {
      return init < x ? x : init;
    }
  };

  template<class Integral>
  __AGENCY_ANNOTATION
  typename std::enable_if<
    !detail::is_tuple_like<Integral>::value,
    Integral
  >::type
    operator()(const Integral& max_size)
  {
    return max_size;
  }

  template<class Tuple>
  __AGENCY_ANNOTATION
  typename std::enable_if<
    detail::is_tuple_like<Tuple>::value,
    size_t
  >::type
    operator()(const Tuple& max_shape_dimensions)
  {
    auto make = maker<array<size_t, std::tuple_size<Tuple>::value>>();

    // recursively turn max_shape_dimensions into a tuple of sizes using this functor
    auto tuple_of_sizes = __tu::tuple_map_with_make(*this, make, max_shape_dimensions);

    // find the maximum over the sizes
    size_t init = 0;
    return __tu::tuple_reduce(tuple_of_sizes, init, max_functor());
  }
};


template<class ShapeTuple>
__AGENCY_ANNOTATION
typename std::enable_if<
  detail::is_tuple_like<ShapeTuple>::value,
  ShapeTuple
>::type
  max_sizes(const ShapeTuple& max_dimensions)
{
  return __tu::tuple_map_with_make(max_sizes_functor(), maker<ShapeTuple>(), max_dimensions);
} // end max_sizes()


template<class Integral>
__AGENCY_ANNOTATION
typename std::enable_if<
  !detail::is_tuple_like<Integral>::value,
  Integral
>::type
  shape_product(const Integral& x)
{
  return x;
} // end shape_product()


template<class Shape>
__AGENCY_ANNOTATION
typename std::enable_if<
  detail::is_tuple_like<Shape>::value,
  size_t
>::type
  shape_product(const Shape& shape)
{
  size_t init = 1;
  return __tu::tuple_reduce(shape, init, [](size_t a, size_t b)
  {
    return a * b;
  });
} // end shape_product()


// there are two overloads for index_space_size()
template<typename Shape>
__AGENCY_ANNOTATION
typename std::enable_if<
  std::is_integral<Shape>::value,
  size_t
>::type
  index_space_size(const Shape& s);

template<typename Shape>
__AGENCY_ANNOTATION
typename std::enable_if<
  !std::is_integral<Shape>::value,
  size_t
>::type
  index_space_size(const Shape& s);


// scalar case
template<typename Shape>
__AGENCY_ANNOTATION
typename std::enable_if<
  std::is_integral<Shape>::value,
  size_t
>::type
  index_space_size(const Shape& s)
{
  return static_cast<size_t>(s);
}

struct index_space_size_functor
{
  template<typename T>
  __AGENCY_ANNOTATION
  size_t operator()(const T& x)
  {
    return index_space_size(x);
  }
};

// tuple case
template<typename Shape>
__AGENCY_ANNOTATION
typename std::enable_if<
  !std::is_integral<Shape>::value,
  size_t
>::type
  index_space_size(const Shape& s)
{
  // transform s into a tuple of sizes
  auto tuple_of_sizes = detail::tuple_map(index_space_size_functor{}, s);

  // reduce the sizes
  return __tu::tuple_reduce(tuple_of_sizes, size_t{1}, [](size_t x, size_t y)
  {
    return x * y;
  });
}


template<class TypeList>
struct merge_first_two_types;

template<class T1, class T2, class... Types>
struct merge_first_two_types<type_list<T1,T2,Types...>>
{
  // XXX we probably want to think carefully about what it means two "merge" two arithmetic tuples together
  template<class U1, class U2>
  using merge_types_t = typename std::common_type<U1,U2>::type;

  using type = type_list<merge_types_t<T1,T2>, Types...>;
};


template<class Shape>
struct merge_front_shape_elements_t_impl
{
  using elements = tuple_elements<Shape>;
  using merged_elements = typename merge_first_two_types<elements>::type;

  // turn the resulting type list into a shape_tuple
  // XXX we should use something like rebind_shape_t<Shape,Types...> here instead
  using shape_tuple_type = type_list_instantiate<shape_tuple, merged_elements>;

  // finally, unwrap single-element tuples
  using type = typename std::conditional<
    (std::tuple_size<shape_tuple_type>::value == 1),
    typename std::tuple_element<0,shape_tuple_type>::type,
    shape_tuple_type
  >::type;
};

template<class Shape>
using merge_front_shape_elements_t = typename merge_front_shape_elements_t_impl<Shape>::type;


template<size_t... Indices,
         class Shape>
__AGENCY_ANNOTATION
merge_front_shape_elements_t<Shape>
  merge_front_shape_elements_impl(detail::index_sequence<Indices...>, const Shape& s)
{
  return merge_front_shape_elements_t<Shape>{agency::get<0>(s) * agency::get<1>(s), agency::get<Indices+2>(s)...};
} // end merge_front_shape_elements_impl()


template<class Shape,
         class = typename std::enable_if<
           detail::is_tuple_like<Shape>::value
         >::type,
         class = typename std::enable_if<
           (std::tuple_size<Shape>::value > 1)
         >::type>
__AGENCY_ANNOTATION
merge_front_shape_elements_t<Shape>
  merge_front_shape_elements(const Shape& s)
{
  constexpr size_t shape_size = std::tuple_size<Shape>::value;

  return detail::merge_front_shape_elements_impl(detail::make_index_sequence<shape_size-2>(), s);
} // merge_front_shape_elements()


// the type of a Shape's head is simply its head element when the Shape is a tuple
// otherwise, it's just the Shape
template<class Shape>
using shape_head_t = typename std::decay<
  decltype(detail::tuple_head_if(std::declval<Shape>()))
>::type;


template<class Shape>
__AGENCY_ANNOTATION
auto shape_head(const Shape& s) -> decltype(detail::tuple_head_if(s))
{
  return detail::tuple_head_if(s);
} // end shape_head()


// the type of a Shape's tail is simply its tail when the Shape is a tuple
// otherwise, it's just an empty tuple
template<class Shape>
using shape_tail_t = decltype(detail::tuple_tail_if(std::declval<Shape>()));

template<class Shape>
__AGENCY_ANNOTATION
shape_tail_t<Shape> shape_tail(const Shape& s)
{
  return detail::tuple_tail_if(s);
} // end shape_tail()


// returns the number of points spanned by a Shape's head element
template<class Shape>
__AGENCY_ANNOTATION
size_t index_space_size_of_shape_head(const Shape& s)
{
  return detail::index_space_size(detail::shape_head(s));
} // end index_space_size_of_shape_head()


template<size_t n, class Shape>
using shape_take_t = detail::decay_t<
  decltype(
    detail::unwrap_single_element_tuple_if(
      detail::tuple_take_if<n>(std::declval<Shape>())
    )
  )
>;


// note that shape_take() unwraps single element tuples which result from tuple_take_if
template<size_t n, class Shape>
__AGENCY_ANNOTATION
shape_take_t<n,Shape> shape_take(const Shape& s)
{
  return detail::unwrap_single_element_tuple_if(detail::tuple_take_if<n>(s));
}


} // end detail
} // agency

