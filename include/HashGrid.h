#pragma once

#include <Eigen/Dense>
#include <vector>
#include <unordered_set>

// Spatial hash grid in 3D.
//
template<typename T>
class HashGrid3d
{
public:

    // Constructor.  
    // Creates a hash grid using the bounding box coordinates [  @a min, @a max  ] 
    // and cell size @a dx.
    //
    HashGrid3d(const Eigen::Vector3f& _min, const Eigen::Vector3f& _max, float _dx) :
        min(_min), max(_max), dx(_dx)
    {
        // Compute the number of cells in each direction x,y,z
        //
        dims[0] = (int) std::ceil((max.x() - min.x()) / _dx);
        dims[1] = (int) std::ceil((max.y() - min.y()) / _dx);
        dims[2] = (int) std::ceil((max.z() - min.z()) / _dx);

        const int size = dims[0] * dims[1] * dims[2];
        bins.resize(size);
    }

    // Returns the cell coordinates of point @a _p.
    // Coordinates [i,j,k] are returned in the array coords[]
    //
    void getCoordinates(const Eigen::Vector3f& _p, int coords[3]) const
    {
        for (int i = 0; i < 3; ++i)
        {
            coords[i] = std::min(dims[i] - 1, std::max(0, (int)std::floor((_p[i] - min[i]) / dx)));
        }
    }

    // Returns true if @a key is valid.
    //
    inline bool isValid(int key) const
    {
        return (0 <= key && key < bins.size());
    }

    // Returns true if cell coordinates [i,j,k] are valid.
    //
    inline bool isValid(int i, int j, int k) const
    {
        return (0 <= i && i < dims[0] && 0 <= j && j < dims[1] && 0 <= k && k < dims[2]);
    }

    // Returns the key corresponding to grid cell with coordinates [i,j,k].
    //
    inline int key(int i, int j, int k) const
    {
        return dims[0] * dims[1] * k + dims[0] * j + i;
    }

    // Accessor for bins.
    // Returns a reference to the bin at cell coordinates [i,j,k]
    //
    std::unordered_set<T>& operator()(int i, int j, int k)
    {
        assert(isValid(i, j, k));

        return bins[key(i, j, k)];
    }

    // Accessor for bins.
    // Returns a reference to the bin using the key.
    //
    std::unordered_set<T>& operator()(int key)
    {
        assert(isValid(key));

        return bins[key];
    }

    // Empty the hash grid.
    //
    void clearBins()
    {
        for (auto& b : bins)
        {
            b.clear();
        }
    }

    std::vector< std::unordered_set<T> > bins;      // The bins.
    Eigen::Vector3f max, min;                       // The bounding box coordinates of the spatial grid.
    int dims[3];                                    // Number of bins in x,y,z
    float dx;                                       // Cell size.

private:

    // Default constructor
    explicit HashGrid3d() : min(-1, -1, -1), max(1, 1, 1), dx(0.1) {}

};