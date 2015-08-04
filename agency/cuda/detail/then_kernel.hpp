#pragma once

#include <memory>

namespace agency
{
namespace cuda
{
namespace detail
{


// XXX WAR nvbug 1671566
struct my_nullptr_t
{
  my_nullptr_t(std::nullptr_t) {}
};


template<class Function>
__global__ void then_kernel(my_nullptr_t, Function f)
{
  f();
}


//template<class Function, class T>
//__global__ void then_kernel(std::nullptr_t, Function f, T* result_ptr)
//{
//  *result_ptr = f();
//}
//
//template<class Function>
//__global__ void then_kernel(std::nullptr_t, Function f, std::nullptr_t)
//{
//  f();
//}


} // end detail
} // end cuda
} // end agency

