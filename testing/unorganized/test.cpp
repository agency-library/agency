#include <agency/agency.hpp>
#include <iostream>
#include <mutex>

int main()
{
  using namespace agency;

  bulk_invoke(seq(4), [&](sequenced_agent &g)
  {
    std::cout << g.index() << std::endl;
  });

  auto f1 = bulk_async(seq(4), [&](sequenced_agent &g)
  {
    std::cout << g.index() << std::endl;
  });

  f1.wait();

  auto f2 = bulk_async(seq(2, seq(1)), [&](sequenced_group<sequenced_agent> &self)
  {
    std::cout << self.index() << std::endl;
  });

  f2.wait();

  auto f3 = bulk_async(seq(3, seq(1, seq(4))), [&](sequenced_group<sequenced_group<sequenced_agent>> &self)
  {
    std::cout << self.index() << std::endl;
  });

  f3.wait();

  return 0;
}

