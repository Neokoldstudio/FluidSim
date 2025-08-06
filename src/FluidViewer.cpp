#include "FluidViewer.h"

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/point_cloud.h"
#include "polyscope/view.h"
#include "imgui.h"

#include <iostream>

using namespace std;

#include "PositionBasedFluid.h"

namespace
{
    static const glm::vec3 highlightColor(1.0f, 0.0f, 0.0f);
    static const glm::vec3 pointColor(0.1f, 0.2f, 1.0f);
    static const float sMaxDensity = 1e4f;

    static inline void randomPointInBox(const AABB& aabb, Eigen::Vector3f& p)
    {
        const Eigen::Vector3f dims = aabb.max - aabb.min;
        p = aabb.min + dims.cwiseProduct(0.5f * Eigen::Vector3f::Random() + Eigen::Vector3f(0.5f, 0.5f, 0.5f));
    }

    static inline void aabbToSurfaceMesh(const AABB& aabb, Eigen::MatrixXf& meshV, Eigen::MatrixXi& meshF)
    {
        meshV.setZero(8, 3);
        meshV.row(0) << aabb.min.x(), aabb.min.y(), aabb.min.z();
        meshV.row(1) << aabb.max.x(), aabb.min.y(), aabb.min.z();
        meshV.row(2) << aabb.max.x(), aabb.min.y(), aabb.max.z();
        meshV.row(3) << aabb.min.x(), aabb.min.y(), aabb.max.z();
        meshV.row(4) << aabb.min.x(), aabb.max.y(), aabb.min.z();
        meshV.row(5) << aabb.max.x(), aabb.max.y(), aabb.min.z();
        meshV.row(6) << aabb.max.x(), aabb.max.y(), aabb.max.z();
        meshV.row(7) << aabb.min.x(), aabb.max.y(), aabb.max.z();
        meshF.setZero(6, 4);
        meshF.row(0) << 0, 3, 2, 1;
        meshF.row(1) << 0, 4, 7, 3;
        meshF.row(2) << 1, 2, 6, 5;
        meshF.row(3) << 4, 5, 6, 7;
        meshF.row(4) << 2, 3, 7, 6;
        meshF.row(5) << 0, 1, 5, 4;
    }

    static inline void fillBox(const AABB& aabb, float rho0, std::vector<Particle>& particles)
    {
        int index = particles.size();
        const float dx = std::pow(1.0f / rho0, 0.333f);
        for (float x = aabb.min.x(); x <= aabb.max.x(); x += dx)
        {
            for (float y = aabb.min.y(); y <= aabb.max.y(); y += dx)
            {
                for (float z = aabb.min.z(); z <= aabb.max.z(); z += dx)
                {
                    particles.push_back(Particle(index++, Eigen::Vector3f(x, y, z), Eigen::Vector3f(0.0f, 0.0f, 0.0f)));
                }
            }
        }
    }
}

FluidViewer::FluidViewer() :
    m_pbf(nullptr),
    m_dt(0.0167f),
    m_paused(true),
    m_stepOnce(false),
    m_fluidPoints(nullptr),
    m_fluidDensity(nullptr)
{

}

FluidViewer::~FluidViewer()
{
}

void FluidViewer::start()
{
    m_pbf = std::make_unique<PositionBasedFluid>();

    // Setup Polyscope
    polyscope::options::programName = "MTI855 Devoir 03 - Fluides";
    polyscope::options::verbosity = 0;
    polyscope::options::usePrefsFile = false;
    polyscope::options::alwaysRedraw = true;
    polyscope::options::ssaaFactor = 2;
    polyscope::options::openImGuiWindowForUserCallback = true;
    //polyscope::options::groundPlaneEnabled = false;
    polyscope::options::automaticallyComputeSceneExtents = true;
    polyscope::options::buildGui = false;
    polyscope::options::maxFPS = -1;
    polyscope::view::windowWidth = 1920;
    polyscope::view::windowHeight = 1080;

    // initialize
    polyscope::init();

    // Specify the update callback
    polyscope::state::userCallback = std::bind(&FluidViewer::draw, this);

    // Create a surface mesh for drawing the bounding box of the fluid
    Eigen::MatrixXf meshV;
    Eigen::MatrixXi meshF;
    aabbToSurfaceMesh(m_pbf->aabb, meshV, meshF);
    polyscope::SurfaceMesh* boxMesh = polyscope::registerSurfaceMesh("box", meshV, meshF);
    boxMesh->setTransparency(0.12f);
    boxMesh->setEdgeWidth(0.1);
    boxMesh->setSurfaceColor(glm::vec3(0.1, 1.0, 0.1));

    createDamBreak();

    // Show the window
    polyscope::show();
}

void FluidViewer::drawGUI()
{
    ImGui::Text("Simulation: ");
    ImGui::Checkbox("Pause", &m_paused);
    if (ImGui::Button("Step once"))
    {
        m_stepOnce = true;
    }

    ImGui::Text("Integration: ");
    ImGui::PushItemWidth(200);
    ImGui::SliderFloat("Time step (dt)", &m_dt, 0.001f, 0.1f, "%0.3f");
    ImGui::PopItemWidth();
    
    ImGui::Text("Simulation params: ");
    ImGui::PushItemWidth(200);
    ImGui::SliderFloat("Kernel radius (h)", &m_pbf->radius, 0.01, 1.0f, "%1.3f");
    ImGui::SliderInt("Density iterations", &m_pbf->maxIter, 1, 100);
    ImGui::SliderFloat("Rest density (rho)", &m_pbf->rho0, 100.0f, sMaxDensity, "%5.1f");
    ImGui::SliderFloat("Vorticity Epsilon", &m_pbf->v_eps, 0.001f, 0.1f, "%0.5f");
    ImGui::SliderFloat("Artifical viscosity (c)", &m_pbf->c, 0.0f, 2e-4f, "%0.5f");
    ImGui::SliderFloat("Artificial pressure strength (k)", &m_pbf->k_corr, 0.0f, 1.0f, "%0.4f");
    ImGui::PopItemWidth();


    ImGui::Text("Scenarios: ");
    ImGui::PushItemWidth(200);
    if (ImGui::Button("Dam Break"))
    {
        createDamBreak();
    }
    if (ImGui::Button("Double Dam Break"))
    {
        createDoubleDamBreak();
    }
    if (ImGui::Button("Droplet"))
    {
        createDroplet();
    }
    ImGui::PopItemWidth();
    
    const unsigned int numParticles = m_pbf->getParticles().size();
    ImGui::LabelText("Number particles", "%d", numParticles);
}

void FluidViewer::initFluid()
{
    const auto& particles = m_pbf->getParticles();
    const unsigned int numParticles = particles.size();

    Eigen::MatrixXf particlesV(numParticles, 3);

    for (int i = 0; i < numParticles; ++i)
    {
        particlesV.row(i) << particles[i].x(0), particles[i].x(1), particles[i].x(2);
    }

    // Register the particles point cloud with Polyscope
    m_fluidPoints = polyscope::registerPointCloud("particles", particlesV);
    m_fluidPoints->setPointRadius(0.005);
    m_fluidPoints->setPointRenderMode(polyscope::PointRenderMode::Sphere);
    m_fluidDensity = m_fluidPoints->addScalarQuantity("density", std::vector<float>(numParticles, 0.0f));
    m_fluidDensity->setEnabled(true);
    m_fluidDensity->setColorMap("viridis");
    m_fluidDensity->setMapRange(std::pair<double, double>(0, sMaxDensity));
}

void FluidViewer::updateFluid()
{
    const auto& particles = m_pbf->getParticles();
    const unsigned int numParticles = particles.size();

    Eigen::MatrixXf particlesV(numParticles, 3);

    for (int i = 0; i < numParticles; ++i)
    {
        particlesV.row(i) << particles[i].x(0), particles[i].x(1), particles[i].x(2);
    }
    // Update fluid point cloud
    m_fluidPoints->updatePointPositions(particlesV);

    std::vector<float> particlesRho(numParticles, 0.0);
    for (const Particle& p : particles)
    {
        particlesRho[p.index] = p.rho;
    }
    m_fluidDensity->updateData(particlesRho);
}

void FluidViewer::draw()
{
    drawGUI();

    // Simulation stepping
    //
    if (!m_paused || m_stepOnce)
    {
        m_pbf->step(m_dt);
        m_stepOnce = false;
    }

    // Update fluid particle positions.
    //
    updateFluid();
}

void FluidViewer::createDoubleDamBreak()
{
    int index = 0;
    std::vector<Particle> particles;

    fillBox(AABB(Eigen::Vector3f(-2.0f, 0.0f, -1.0f), Eigen::Vector3f(-1.6f, 2.0f, 1.0f)), m_pbf->rho0, particles);
    fillBox(AABB(Eigen::Vector3f(1.6f, 0.0f, -1.0f), Eigen::Vector3f(2.0f, 2.0f, 1.0f)), m_pbf->rho0, particles);

    m_pbf->setParticles(particles);
    initFluid();
}


void FluidViewer::createDamBreak()
{
    int index = 0;
    std::vector<Particle> particles;

    AABB aabb(Eigen::Vector3f(-2.0f, 0.0f, -1.0f), Eigen::Vector3f(-1.8f, 2.0f, 1.0f));
    fillBox(aabb, m_pbf->rho0, particles);

    m_pbf->setParticles(particles);
    initFluid();
}


void FluidViewer::createDroplet()
{
    int index = 0;
    std::vector<Particle> particles;

    AABB aabb(Eigen::Vector3f(-0.1f, 0.8f, -0.1f), Eigen::Vector3f(0.1f, 1.2f, 0.1f));
    fillBox(aabb, m_pbf->rho0, particles);

    m_pbf->setParticles(particles);
    initFluid();
}

