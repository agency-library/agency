#include <agency/agency.hpp>
#include <random>
#include <cstdio>

void test()
{
  using namespace agency;

  std::default_random_engine rng(13);

  {
    int shape0 = rng() % 10;

    int reference_rank = 0; 
    for(int i = 0; i < shape0; ++i)
    {
      int index = i;
      int shape = shape0;

      int rank = agency::detail::index_cast<int>(index, shape, shape);

      assert(rank == reference_rank);
      ++reference_rank;
    }
  }

  {
    int shape0 = rng() % 10;
    int shape1 = rng() % 10;

    int reference_rank = 0; 
    for(int i = 0; i < shape0; ++i)
    {
      for(int j = 0; j < shape1; ++j)
      {
        int2 index{i,j};
        int2 shape{shape0,shape1};

        int rank = agency::detail::index_cast<int>(index, shape, shape[0] * shape[1]);

        assert(rank == reference_rank);
        ++reference_rank;
      }
    }
  }

  {
    int shape0 = rng() % 10;
    int shape1 = rng() % 10;
    int shape2 = rng() % 10;

    int reference_rank = 0; 
    for(int i = 0; i < shape0; ++i)
    {
      for(int j = 0; j < shape1; ++j)
      {
        for(int k = 0; k < shape2; ++k)
        {
          int3 index{i,j,k};
          int3 shape{shape0,shape1,shape2};

          int rank = agency::detail::index_cast<int>(index, shape, shape[0] * shape[1] * shape[2]);

          assert(rank == reference_rank);
          ++reference_rank;
        }
      }
    }
  }

  {
    int shape0 = rng() % 10;
    int shape1 = rng() % 10;
    int shape2 = rng() % 10;
    int shape3 = rng() % 10;

    int reference_rank = 0; 
    for(int i = 0; i < shape0; ++i)
    {
      for(int j = 0; j < shape1; ++j)
      {
        for(int k = 0; k < shape2; ++k)
        {
          for(int l = 0; l < shape3; ++l)
          {
            int4 index{i,j,k,l};
            int4 shape{shape0,shape1,shape2,shape3};

            int rank = agency::detail::index_cast<int>(index, shape, shape[0] * shape[1] * shape[2] * shape[3]);

            assert(rank == reference_rank);
            ++reference_rank;
          }
        }
      }
    }
  }
}

int main()
{
  test();

  std::cout << "OK" << std::endl;

  return 0;
}

