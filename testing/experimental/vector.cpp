#include <iostream>
#include <cassert>
#include <agency/experimental/vector.hpp>

void test_fill_construct()
{
  using namespace agency::experimental;

  {
    // test fill construct empty vector
    vector<int> v(0, 13);

    assert(v.empty());
  }
}

int main()
{
  test_fill_construct();

  std::cout << "OK" << std::endl;

  return 0;
}

