#include "PositionBasedFluid.h"
#include "Eigen/src/Core/Matrix.h"
#include "glm/common.hpp"
#include <omp.h>

namespace
{
    static const float pi = (3.14159265358979323846264338327950288f);

    const Eigen::Vector3f gravity(0.0f, -GRAVITY, 0.0f);

    // SPH kernel implementations
    //
    class Kernel
    {
    private:
        float h, h2, h3, h9;
        float deltaqSq;
        float kPoly, kSpiky;
        explicit Kernel() : h(0.0f), h2(0.0f), h3(0.0f), h9(0.0f), kPoly(0.0), kSpiky(0.0), deltaqSq(0.0f)
        {

        }

    public:
        Kernel(float _h) : h(_h), h2(h* h), h3(h2* h), h9(h3* h3* h3), deltaqSq(0.1f*h*h)
        {
            kPoly = 315.0f / (64.0f * pi * h9);
            kSpiky = 45.0f / (pi * h3 * h3);
        }

        // The poly-6 kernel.
        float poly6(const Eigen::Vector3f& pi, const Eigen::Vector3f& pj) const
        {
            const Eigen::Vector3f pipj = pi - pj;
            const float r = pipj.norm();
            if (r < h)
            {
                const float x = h2 - r*r;
                return kPoly * x * x * x;
            }
            return 0.0f;
        }

        // The "spiky" kernel gradient. Use this for gradient calculations, e.g. when computing lambda 
        //
        Eigen::Vector3f spiky(const Eigen::Vector3f& pi, const Eigen::Vector3f& pj)
        {
            const Eigen::Vector3f pipj = pi - pj;
            const float r = pipj.norm();

            if( r < 1e-6f )
                return Eigen::Vector3f::Zero();
            else if (r < h)
            {
                const float x = (h - r);
                return -((kSpiky * x * x) / r) * pipj;
            }

            return Eigen::Vector3f::Zero();
        }

        // Kernel used for the denominator of the scorr coefficient Eq. 11.
        float scorr()
        {
            const float x = h2 - deltaqSq;
            return kPoly * x * x * x;
        }
    };

}

PositionBasedFluid::PositionBasedFluid() : m_hashGrid(aabb.min, aabb.max, radius), m_particles(),
rho0(6400.0f), eps(400.0f), aabb(Eigen::Vector3f(-2.0f, 0.0f, -1.0f), Eigen::Vector3f(2.0f, 2.0f, 1.0f)),
maxIter(4), v_eps(0.001), c(2e-4f), k_corr(2e-4f), radius(0.1f)
{

}

void PositionBasedFluid::setParticles(const std::vector<Particle>& _particles)
{
    m_particles = _particles;  
    m_hashGrid.clearBins();
    updateHashGrid();
}

void PositionBasedFluid::step(float dt)
{
    // TODO
    // Compute predicted particle positions
    // (see lines 1 to 3, Algorithm 1)
    //
    for (Particle& p : m_particles)
    {
		p.v += gravity * dt; // Apply gravity to the velocity.
        p.xstar = p.x + p.v * dt;
    }

    updateHashGrid();

    // TODO 
    // Find neighboring particles
    // (see lines 5 to 7, Algorithm 1)
    //
    buildNeighborhood();

    const unsigned int numParticles = m_particles.size();

    Kernel W(radius);

    for (int iter = 0; iter < maxIter; ++iter)  // density solve start.
    {
        // TODO Compute the density forces
        //
        // (see lines 9-11, Algorithm 1)
        //
        #pragma omp parallel for
        for (int i = 0; i < numParticles; ++i)
        {
            // TODO
            // Compute the particle density using the poly-6 kernel 
            // to integrate the mass of nearby particles.
            //
            Particle &p = m_particles[i];
            p.rho = 0.0f;
            for (Particle* neighbor : p.N)
            {
                if (neighbor) // Ensure neighbor pointer is valid
                {
                    p.rho += W.poly6(p.xstar, neighbor->xstar); // according to Macklin and Müller, we treat the particle mass as 1.0f
                }
            }

            // TODO
            // Compute the density constraint lambda using Equation 11.
            //
            float C_i  = (p.rho / rho0) - 1.0f;

            Eigen::Vector3f sumGrad = Eigen::Vector3f::Zero();
            float sumNrmSq = 0.0f;
            for(Particle* neighbor : p.N) {
                if (neighbor != &p) // k != i
                {
                    Eigen::Vector3f grad = (1.0f / rho0) * W.spiky(p.xstar, neighbor->xstar);
                    sumGrad += grad;
                    sumNrmSq += grad.squaredNorm(); 
                }
            }

            p.lambda = -C_i / (sumGrad.dot(sumGrad) + sumNrmSq + eps);
        }

        // TODO
        // Compute delta p position updates for all particles
        //   using Equation 12.
        //  (see lines 12-14, Algorithm 1)
        //
        // TODO 
        // Compute the tensile instability correction using Equation 13 
        //   and revise the code for computing delta p according to Equation 14.
        //
        #pragma omp parallel for
        for(int i = 0; i < numParticles; ++i)
        {
            Particle &p = m_particles[i];

            p.dp = Eigen::Vector3f::Zero();
            for (Particle* neighbor : p.N)
            {
                float Scorr = -k_corr * std::pow(W.poly6(p.xstar , neighbor->xstar) / W.scorr(), 4);
                p.dp += (p.lambda + neighbor->lambda + Scorr) * W.spiky(p.xstar, neighbor->xstar);
            }
            p.dp /= rho0;
        }

        // TODO
        // Update the position for all particles
        // (see lines 16-18, Algorithm 1)
        //
        #pragma omp parallel for
        for (int i = 0; i < numParticles; ++i)
        {
            Particle& p = m_particles[i];
            p.xstar += p.dp; // Update the intermediate position.
        }

        // TODO
        // Perform collision detection after each position update, 
        // and project particle that have moved outside to the 
        // inside of the bounding box.  
        // 
        // This requires implementing the boxCollision() function.
        //
        boxCollision();

    }   // density solve end.


    // TODO
    // Perform a final position update using most recent xstar.z
    //
    #pragma omp parallel for
    for (int i = 0; i < numParticles; ++i)
    {
        Particle& p = m_particles[i];
        p.v = (p.xstar - p.x)/dt; // Update the velocity based on the position change.
        p.x = p.xstar; // Update the position to the intermediate position.
    }

    // TODO
    // Viscosity computation.
    // Accumulate velocity differences between neighboring particles and store in Particle::vdiff
    //
    #pragma omp parallel for
    for (int i = 0; i < numParticles; ++i)
    {
        Particle& p = m_particles[i];
		p.vdiff = Eigen::Vector3f::Zero();

        for (Particle* neighbor : p.N)
        {
            if (neighbor != &p) // k != i
            {
               p.vdiff += (neighbor->v - p.v) * W.poly6(p.xstar, neighbor->xstar);
            }
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < numParticles; ++i)
    {
        Particle& p = m_particles[i];
        p.v += c * p.vdiff;
    }

    // 1. Compute vorticity vector (curl)
    #pragma omp parallel for
    for (int i = 0; i < numParticles; ++i)
    {
        Particle& p = m_particles[i];
        p.omega = Eigen::Vector3f::Zero();

        for (Particle* neighbor : p.N)
        {
            if (neighbor != &p)
            {
                Eigen::Vector3f vij = neighbor->v - p.v;
                p.omega += vij.cross(W.spiky(p.xstar, neighbor->xstar));
            }
        }
    }

    // 2. Compute vorticity gradient (η) and vorticity force
    #pragma omp parallel for
    for (int i = 0; i < numParticles; ++i)
    {
        Particle& p = m_particles[i];

        Eigen::Vector3f eta = Eigen::Vector3f::Zero();
        for (Particle* neighbor : p.N)
        {
            if (neighbor != &p)
            {
                Eigen::Vector3f grad = W.spiky(p.xstar, neighbor->xstar);
                float w = grad.norm();
                eta += (neighbor->omega.norm()) * grad; // Gradient of vorticity magnitude
            }
        }

        if (eta.norm() > 1e-5)
        {
            Eigen::Vector3f N = eta.normalized();
            Eigen::Vector3f fVort = v_eps * N.cross(p.omega);
            p.v += dt * fVort;
        }
    }
}

void PositionBasedFluid::updateHashGrid()
{
    int coords[3];
    const unsigned int numParticles = m_particles.size();
    for (int i = 0; i < numParticles; ++i)
    {
        Particle& p = m_particles[i];
        m_hashGrid.getCoordinates(p.xstar, coords);
        const int key = m_hashGrid.key(coords[0], coords[1], coords[2]);

        // If the hash key has changed, 
        // remove the particle from its current cell and add it to the new one.
        //
        if ( key != p.key)
        {
            if (m_hashGrid.isValid(p.key))
            {
                m_hashGrid(p.key).erase(&p);
            }
            m_hashGrid(key).insert(&p);
            p.key = key;
        }
    }
}

void PositionBasedFluid::buildNeighborhood()
{
    const unsigned int numParticles = m_particles.size();

    #pragma omp parallel for
    for (int i = 0; i < numParticles; ++i)
    {
        Particle& p = m_particles[i];
        p.N.clear();
        // TODO
        // Build the neighbor list for each particle.
        //
        // Start by computing the cell coordinates for the current particle,
        //   e.g.  m_grid.getCoordinates(p.xstar, coords);
        //
        // Then, use the spatial hash grid to access lists of particles in
        // nearby grid cells. Note : there are 9 grid cells
        //
        // Finally, for particles within distance @a radius of the current
        // particle @a p,
        //  add it to the list of neighbors
        int coords[3];
        m_hashGrid.getCoordinates(p.xstar, coords);
        for (int x = coords[0] - 1; x <= coords[0] + 1; ++x)
        {
            for (int y = coords[1] - 1; y <= coords[1] + 1; ++y)
            {
                for (int z = coords[2] - 1; z <= coords[2] + 1; ++z)
                {
                    int key = m_hashGrid.key(x, y, z);
                    if (m_hashGrid.isValid(key))
                    {
                        auto& cellParticles = m_hashGrid(key);
                        for (Particle* neighbor : cellParticles)
                        {
                            if ((p.xstar - neighbor->xstar).squaredNorm() < radius * radius)
                            {
                                p.N.push_back(neighbor);
                            }
                        }
                    }
                }
            }
        }
    }
}

namespace {
    // Custom clamp function for keeping the particles inside of the bouding box, assuming min and max of the aabb are in world space.
    Eigen::Vector3f clampVec3(const Eigen::Vector3f& v, const Eigen::Vector3f& min, const Eigen::Vector3f& max) {
        return Eigen::Vector3f(
            std::max(min.x(), std::min(max.x(), v.x())),
            std::max(min.y(), std::min(max.y(), v.y())),
            std::max(min.z(), std::min(max.z(), v.z()))
        );
    }
}

void PositionBasedFluid::boxCollision()
{
    const unsigned int numParticles = m_particles.size();

    // TODO 
    // Loop over all particles and move them inside the
    // bounding box @a aabb if they are outside
    //
    #pragma omp parallel for
    for (int i = 0; i < numParticles; ++i)
    {
        Particle& p = m_particles[i];

        p.xstar = clampVec3(p.xstar, aabb.min, aabb.max);
    }
}

