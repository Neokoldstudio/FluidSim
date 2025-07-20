#pragma once

#include <Eigen/Dense>
#include <vector>

namespace polyscope
{
    class PointCloud;
    class PointCloudScalarQuantity;
}

class PositionBasedFluid;

// Viewer for a soft body simulation.
//
class FluidViewer
{
public:
    FluidViewer();
    virtual ~FluidViewer();

    void start();

private:
    void createDoubleDamBreak();
    void createDamBreak();
    void createDroplet();

    void initFluid();
    void updateFluid();
    void draw();
    void drawGUI();

    polyscope::PointCloud* m_fluidPoints;
    polyscope::PointCloudScalarQuantity* m_fluidDensity;

    std::unique_ptr<PositionBasedFluid> m_pbf;

    // Simulation parameters
    float m_dt;                         //< Time step parameter.
    bool m_paused;                      //< Pause the simulation.
    bool m_stepOnce;                    //< Advance the simulation by one frame and then stop.
};
