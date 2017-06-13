#include <iostream>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <vector>
#include <list>
#include <agency/container/vector.hpp>

void test_assign_operator()
{
  using namespace agency;

  {
    // test assign empty to empty

    vector<int> empty;
    vector<int> other_empty;

    empty = other_empty;

    assert(empty.empty());
    assert(other_empty.empty());
  }

  {
    // test assign non-empty to empty

    vector<int> initially_empty;
    vector<int> not_empty(10, 13);

    initially_empty = not_empty;

    assert(!initially_empty.empty());
    assert(!not_empty.empty());
    assert(initially_empty == not_empty);
  }

  {
    // test assign non-empty to non-empty
    
    vector<int> non_empty(100, 7);
    vector<int> other_not_empty(10, 13);

    non_empty = other_not_empty;

    assert(!non_empty.empty());
    assert(!other_not_empty.empty());
    assert(non_empty == other_not_empty);
  }
}

int main()
{
  test_assign_operator();

  return 0;
}

