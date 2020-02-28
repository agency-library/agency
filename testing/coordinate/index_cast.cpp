#include <agency/agency.hpp>
#include <random>
#include <cstdio>

void test()
{
  {
    // 1D -> 2D
    
    agency::uint2 to_shape{8,2};
    unsigned int from_shape = 16;

    size_t reference_rank = 0;
    for(unsigned int from_idx = 0; from_idx < from_shape; ++from_idx)
    {
      agency::uint2 to_idx = agency::detail::index_cast<agency::uint2>(from_idx, from_shape, to_shape);
      size_t rank = agency::detail::colexicographic_rank(to_idx, to_shape);

      assert(reference_rank == rank);
      ++reference_rank;
    }
  }

  {
    // 2D -> 1D

    unsigned int to_shape{16};
    agency::uint2 from_shape{2,8};

    size_t reference_rank = 0;
    for(unsigned int i = 0; i < from_shape[0]; ++i)
    {
      for(unsigned int j = 0; j < from_shape[1]; ++j)
      {
        agency::uint2 from_idx{i,j};

        unsigned int to_idx = agency::detail::index_cast<unsigned int>(from_idx, from_shape, to_shape);
        size_t rank = agency::detail::colexicographic_rank(to_idx, to_shape);

        assert(reference_rank == rank);
        ++reference_rank;
      }
    }
  }


  {
    // 2D -> 3D

    using namespace agency;
    uint2 from_shape{4,2};
    uint3 to_shape{4,1,2};

    size_t reference_rank = 0;
    for(unsigned int i = 0; i < from_shape[0]; ++i)
    {
      for(unsigned int j = 0; j < from_shape[1]; ++j)
      {
        agency::uint2 from_idx{i,j};

        agency::uint3 to_idx = agency::detail::index_cast<agency::uint3>(from_idx, from_shape, to_shape);
        size_t rank = agency::detail::colexicographic_rank(to_idx, to_shape);

        assert(reference_rank == rank);
        ++reference_rank;
      }
    }
  }


  {
    // 3D -> 2D

    using namespace agency;
    uint2 to_shape{2,4};
    uint3 from_shape{2,4,1};

    size_t reference_rank = 0;
    for(unsigned int i = 0; i < from_shape[0]; ++i)
    {
      for(unsigned int j = 0; j < from_shape[1]; ++j)
      {
        for(unsigned int k = 0; k < from_shape[2]; ++k)
        {
          agency::uint3 from_idx{i,j,k};

          agency::uint2 to_idx = agency::detail::index_cast<agency::uint2>(from_idx, from_shape, to_shape);
          size_t rank = agency::detail::colexicographic_rank(to_idx, to_shape);

          assert(reference_rank == rank);
          ++reference_rank;
        }
      }
    }
  }


  {
    // 3D -> 1D (with random inputs)
    using namespace agency;
    std::default_random_engine rng;

    uint3 from_shape(rng()%10, rng()%10, rng()%10);
    unsigned int to_shape = from_shape[0] * from_shape[1] * from_shape[2];

    size_t reference_rank = 0; 
    for(unsigned int i = 0; i < from_shape[0]; ++i)
    {
      for(unsigned int j = 0; j < from_shape[1]; ++j)
      {
        for(unsigned int k = 0; k < from_shape[2]; ++k)
        {
          uint3 from_idx{i,j,k};
          unsigned int to_idx = agency::detail::index_cast<unsigned int>(from_idx, from_shape, to_shape);
          size_t rank = agency::detail::colexicographic_rank(to_idx, to_shape);

          assert(reference_rank == rank);
          ++reference_rank;
        }
      }
    }
  }


  {
    // 1D -> 1D
    std::default_random_engine rng;

    unsigned int from_shape = rng() % 10; 
    int to_shape = from_shape;

    size_t reference_rank = 0; 
    for(unsigned int from_idx = 0; from_idx < from_shape; ++from_idx)
    {
      int to_idx = agency::detail::index_cast<int>(from_idx, from_shape, to_shape);
      size_t rank = agency::detail::colexicographic_rank(to_idx, to_shape);

      assert(reference_rank == rank);
      ++reference_rank;
    }
  }


  {
    // 2D -> 1D
    std::default_random_engine rng;

    agency::int2 from_shape(rng() % 10, rng() % 10);
    int to_shape = from_shape[0] * from_shape[1];

    size_t reference_rank = 0; 
    for(int i = 0; i < from_shape[0]; ++i)
    {
      for(int j = 0; j < from_shape[1]; ++j)
      {
        agency::int2 from_idx{i,j};

        int to_idx = agency::detail::index_cast<int>(from_idx, from_shape, to_shape);
        size_t rank = agency::detail::colexicographic_rank(to_idx, to_shape);

        assert(reference_rank == rank);
        ++reference_rank;
      }
    }
  }

  {
    // 3D -> 1D
    std::default_random_engine rng;

    agency::int3 from_shape(rng() % 10, rng() % 10, rng() % 10);
    int to_shape = from_shape[0] * from_shape[1] * from_shape[2];

    size_t reference_rank = 0; 
    for(int i = 0; i < from_shape[0]; ++i)
    {
      for(int j = 0; j < from_shape[1]; ++j)
      {
        for(int k = 0; k < from_shape[2]; ++k)
        {
          agency::int3 from_idx{i,j,k};

          int to_idx = agency::detail::index_cast<int>(from_idx, from_shape, to_shape);
          size_t rank = agency::detail::colexicographic_rank(to_idx, to_shape);

          assert(reference_rank == rank);
          ++reference_rank;
        }
      }
    }
  }

  {
    // 4D -> 1D
    std::default_random_engine rng;

    agency::int4 from_shape(rng() % 10, rng() % 10, rng() % 10, rng() % 10);
    int to_shape = from_shape[0] * from_shape[1] * from_shape[2] * from_shape[3];

    size_t reference_rank = 0; 
    for(int i = 0; i < from_shape[0]; ++i)
    {
      for(int j = 0; j < from_shape[1]; ++j)
      {
        for(int k = 0; k < from_shape[2]; ++k)
        {
          for(int l = 0; l < from_shape[3]; ++l)
          {
            agency::int4 from_idx{i,j,k,l};

            int to_idx = agency::detail::index_cast<int>(from_idx, from_shape, to_shape);
            size_t rank = agency::detail::colexicographic_rank(to_idx, to_shape);

            assert(reference_rank == rank);
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

