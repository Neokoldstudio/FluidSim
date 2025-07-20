#pragma once

#include "HashGrid.h"
#include <Eigen/Dense>
#include <vector>
#include <list>

// Fluid particle.
//
struct Particle
{
    Particle(int _index, const Eigen::Vector3f& _x, const Eigen::Vector3f& _v) : 
        rho(0.0f), x(_x), xstar(_x), v(_v), f(0.0f, 0.0f, 0.0f), 
        dp(0.0f, 0.0f, 0.0f), lambda(0.0f), N(), key(-1), index(_index) { }
    
    std::list<Particle*> N;     // List of neighbors
    Eigen::Vector3f x;          // Position
    Eigen::Vector3f v;          // Velocity
    Eigen::Vector3f f;          // External forces

    int key;                    // Key for spatial hashing (default value = -1 invalid)
    int index;                  // Particle id

    // Auxiliary variables
    //
    Eigen::Vector3f xstar;      // Intermediate position
    Eigen::Vector3f dp;         // Position update
    Eigen::Vector3f vdiff;      // Velocity difference
    float lambda;               // Density "force"
    float rho;                  // Fluid density at the particle

};

// Axis-aligned bounding box (AABB)
//
struct AABB
{
    AABB(const Eigen::Vector3f& _min, const Eigen::Vector3f& _max) : min(_min), max(_max) { }
    Eigen::Vector3f min, max;
};

// An implementation of "Position Based Fluids" by Macklin and Mueller (2013).
// 
class PositionBasedFluid
{
public:
    float rho0;         // Rest density.
    float eps;          // Relaxation parameter.
    AABB aabb;          // Axis-aligned bounding box used for collision handling.
    int maxIter;        // Density solve iterations.
    float c;            // Viscosity coefficient.
    float k_corr;       // Strength of tension correction force. 
    float radius;       // Kernel radius.

    // Constructor.
    PositionBasedFluid();

    // Advance the simulation by time step @a _dt.
    void step(float _dt);

    // Accessors and mutators for fluid particles.
    void setParticles(const std::vector<Particle>& _particles);
    const std::vector<Particle>& getParticles() const { return m_particles; }
    std::vector<Particle>& getParticles() { return m_particles; }

private:

    void updateHashGrid();
    void buildNeighborhood();
    void boxCollision();

    std::vector<Particle> m_particles;
    HashGrid3d<Particle*> m_hashGrid;

};