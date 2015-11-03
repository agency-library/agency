#include <agency/cuda/future.hpp>
#include <cassert>

int main()
{
  {
    agency::cuda::detail::asynchronous_state<int> f1(0, 13);
    agency::cuda::detail::asynchronous_state<int> f2(0, 7);

    std::cout << "pointers: " << f1.data() << ", " << f2.data() << std::endl;

    agency::cuda::detail::asynchronous_state_tuple<int,int> f3(f1,f2);

    assert(f3.valid());

    auto pointer_tuple = f3.data();

    assert(f3.valid());

    std::cout << "pointers: " << agency::detail::get<0>(pointer_tuple) << ", " << agency::detail::get<1>(pointer_tuple) << std::endl;

    auto ref_tuple = *pointer_tuple;

    auto tuple = f3.get();

    assert(!f3.valid());

    std::cout << "tuple: " << agency::detail::get<0>(tuple) << ", " << agency::detail::get<1>(tuple) << std::endl;
  }

  std::cout << "OK" << std::endl;

  return 0;
}

