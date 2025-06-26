/*------------------------------------------------------ 
This file is part of the LAMMPS-LBM-PSM project.

LAMMPS-LBM-PSM is an open-source project distributed
under the GNU General Public License.

See the README and License file in the top-level 
LAMMPS-LBM-PSM directory for more details.

Tim Najuch, 2022
------------------------------------------------------*/

#include "LBM_PSM_exchangeParticleData.h"
#include <cmath>
#include <algorithm>

ExchangeParticleData::ExchangeParticleData(int dimension_, const std::vector<double>& origin_) : 
    dimension(dimension_),
    origin(origin_),
    sqrt2half_2D(std::sqrt(2.1)/2.0),
    sqrt2half_3D(std::sqrt(3.1)/2.0)
{}

void ExchangeParticleData::setParticlesOnLattice(LBMPSMLattice *lattice_, 
                                                 UnitConversion *unitConversion, 
                                                 int numberParticles, 
                                                 LAMMPS_NS::tagint *tag, 
                                                 double **xPart, double **uPart, 
                                                 double **omega, double *rp)
{
    const std::vector<int> procCoords = lattice_->get_procCoordinates();
    const int envWidth = lattice_->get_envelopeWidth();
    const int nx = lattice_->get_nx();
    const int ny = lattice_->get_ny();
    const int nz = lattice_->get_nz();
    const int ny_nz = ny * nz;
    
    // Precompute domain offsets - FIXED: use original calculation
    double offset_x = 0.0;
    for(int iproc = 0; iproc < procCoords[0]; iproc++) {
        offset_x += lattice_->get_nxLocal(iproc);
    }
    offset_x = offset_x - (procCoords[0] * 2.0 * envWidth) - 1;
    
    double offset_y = 0.0;
    for(int jproc = 0; jproc < procCoords[1]; jproc++) {
        offset_y += lattice_->get_nyLocal(jproc);
    }
    offset_y = offset_y - (procCoords[1] * 2.0 * envWidth) - 1;
    
    double offset_z = 0.0;
    if (dimension == 3) {
        for(int kproc = 0; kproc < procCoords[2]; kproc++) {
            offset_z += lattice_->get_nzLocal(kproc);
        }
        offset_z = offset_z - (procCoords[2] * 2.0 * envWidth) - 1;
    }

    // Precompute constants for velocity calculation
    const double freqFactor = unitConversion->get_freq_lb(1.0);
    const double velFactor = unitConversion->get_vel_lb(1.0);
    
    for(int iPart = 0; iPart < numberParticles; ++iPart) {
        // Convert particle position to lattice units
        const double x_lb_global = unitConversion->get_pos_lb(xPart[iPart][0]-origin[0]);
        const double y_lb_global = unitConversion->get_pos_lb(xPart[iPart][1]-origin[1]);
        const double z_lb_global = unitConversion->get_pos_lb(xPart[iPart][2]-origin[2]);
        
        // Convert to local processor coordinates - FIXED: use original offset calculation
        const double x_lb_local = x_lb_global - offset_x;
        const double y_lb_local = y_lb_global - offset_y;
        const double z_lb_local = (dimension == 3) ? (z_lb_global - offset_z) : 0.0;
        
        const double r_lb = unitConversion->get_radius_lb(rp[iPart]);
        const int nodeZoneExtension = 5;
        
        // Calculate bounding box - FIXED: use integer casting like original
        int x_min = static_cast<int>(x_lb_local - r_lb) - nodeZoneExtension;
        int x_max = static_cast<int>(x_lb_local + r_lb) + nodeZoneExtension;
        int y_min = static_cast<int>(y_lb_local - r_lb) - nodeZoneExtension;
        int y_max = static_cast<int>(y_lb_local + r_lb) + nodeZoneExtension;
        int z_min = 0;
        int z_max = 1;
        if (dimension == 3) {
            z_min = static_cast<int>(z_lb_local - r_lb) - nodeZoneExtension;
            z_max = static_cast<int>(z_lb_local + r_lb) + nodeZoneExtension;
        }
        
        // Clamp to domain bounds
        x_min = std::max(0, std::min(x_min, nx-1));
        x_max = std::max(0, std::min(x_max, nx-1));
        y_min = std::max(0, std::min(y_min, ny-1));
        y_max = std::max(0, std::min(y_max, ny-1));
        z_min = std::max(0, std::min(z_min, nz-1));
        z_max = std::max(0, std::min(z_max, nz-1));
        
        // Precompute particle velocity in lattice units
        const double u_lb[3] = {
            uPart[iPart][0] * velFactor,
            uPart[iPart][1] * velFactor,
            uPart[iPart][2] * velFactor
        };
        
        // Precompute angular velocity factors
        const double omega_lb[3] = {
            omega[iPart][0] * freqFactor,
            omega[iPart][1] * freqFactor,
            omega[iPart][2] * freqFactor
        };
        
        // Process nodes in the bounding box
        for(int i = x_min; i <= x_max; ++i) {
            const double dx = static_cast<double>(i) - x_lb_local;
            
            for(int j = y_min; j <= y_max; ++j) {
                const double dy = static_cast<double>(j) - y_lb_local;
                const int base_ij = i * ny_nz + j * nz;
                
                for(int k = z_min; k <= z_max; ++k) {
                    const double dz = (dimension == 3) ? (static_cast<double>(k) - z_lb_local) : 0.0;
                    const int ind_phys_1D = base_ij + k;
                    
                    // Calculate solid fraction
                    const double sf = calcSolidFraction(i, j, k, x_lb_local, y_lb_local, z_lb_local, r_lb);
                    
                    // Get existing particle data
                    const auto& nodeData = lattice_->getParticleDataOnLatticeNode(ind_phys_1D);
                    LAMMPS_NS::tagint id_old = -1;
                    double sf_old = 0.0;
                    
                    if (tag[iPart] == nodeData.particleID[0]) {
                        id_old = nodeData.particleID[0];
                        sf_old = nodeData.solidFraction[0];
                    } else if (tag[iPart] == nodeData.particleID[1]) {
                        id_old = nodeData.particleID[1];
                        sf_old = nodeData.solidFraction[1];
                    }
                    
                    // Calculate node velocity with rotation - FIXED: use actual dx, dy, dz
                    double uNode[3] = {u_lb[0], u_lb[1], u_lb[2]};
                    uNode[0] += omega_lb[1]*dz - omega_lb[2]*dy;
                    uNode[1] += omega_lb[2]*dx - omega_lb[0]*dz;
                    uNode[2] += omega_lb[0]*dy - omega_lb[1]*dx;
                    
                    // REVERTED TO ORIGINAL UPDATE LOGIC
                    const int decFlag = (sf > 0.00001) + 2*(sf_old > 0.00001);
                    switch(decFlag) {
                        case 0: // sf == 0 && sf_old == 0
                            lattice_->setToZero(ind_phys_1D, tag[iPart]);
                            break;
                        case 1: // sf > 0 && sf_old == 0
                            lattice_->setParticleOnLattice(ind_phys_1D, tag[iPart], uNode, sf);
                            break;
                        case 2: // sf == 0 && sf_old > 0
                            if(id_old == tag[iPart]) 
                                lattice_->setToZero(ind_phys_1D, tag[iPart]);
                            break;
                        case 3: // sf > 0 && sf_old > 0
                            if(sf > sf_old || id_old == tag[iPart])
                                lattice_->setParticleOnLattice(ind_phys_1D, tag[iPart], uNode, sf);
                            break;
                    }
                }
            }
        }
    }
}

double ExchangeParticleData::calcSolidFraction(int i, int j, int k, 
                                              double xP_LB, double yP_LB, 
                                              double zP_LB, double rP_LB) const
{
    const int slicesPerDim = 10;
    const double sliceWidth = 1.0 / slicesPerDim;
    const double rSq = rP_LB * rP_LB;
    const double sqrt2half = (dimension == 3) ? sqrt2half_3D : sqrt2half_2D;
    
    // Calculate actual distance - FIXED: use non-squared distance
    const double dx = static_cast<double>(i) - xP_LB;
    const double dy = static_cast<double>(j) - yP_LB;
    const double dz = (dimension == 3) ? (static_cast<double>(k) - zP_LB) : 0.0;
    const double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    
    // Early exit for completely inside or outside
    const double rP = rP_LB + sqrt2half;
    if (dist > rP) return 0.0;
    
    const double rM = rP_LB - sqrt2half;
    if (dist < rM) return 1.0;
    
    // Subgrid sampling
    double dx_sq[slicesPerDim], dy_sq[slicesPerDim], dz_sq[slicesPerDim];
    for (int idx = 0; idx < slicesPerDim; ++idx) {
        const double delta = (idx + 0.5) * sliceWidth - 0.5;
        dx_sq[idx] = (dx + delta) * (dx + delta);
        dy_sq[idx] = (dy + delta) * (dy + delta);
        if (dimension == 3) {
            dz_sq[idx] = (dz + delta) * (dz + delta);
        }
    }
    
    // Count subgrid points inside particle
    int count = 0;
    if (dimension == 2) {
        for (int i = 0; i < slicesPerDim; ++i) {
            for (int j = 0; j < slicesPerDim; ++j) {
                if (dx_sq[i] + dy_sq[j] < rSq) ++count;
            }
        }
        return static_cast<double>(count) / (slicesPerDim * slicesPerDim);
    } 
    else {
        for (int i = 0; i < slicesPerDim; ++i) {
            for (int j = 0; j < slicesPerDim; ++j) {
                for (int k = 0; k < slicesPerDim; ++k) {
                    if (dx_sq[i] + dy_sq[j] + dz_sq[k] < rSq) ++count;
                }
            }
        }
        return static_cast<double>(count) / (slicesPerDim * slicesPerDim * slicesPerDim);
    }
}

void ExchangeParticleData::calculateHydrodynamicInteractions(LBMPSMLattice *lattice_, 
                                                            UnitConversion *unitConversion, 
                                                            LAMMPS_NS::tagint tag, 
                                                            double *xPart, double rp, 
                                                            std::vector<double> &fHydro, 
                                                            std::vector<double> &tHydro, 
                                                            std::vector<double> &stresslet)
{
    // Get processor coordinates as vector
    const std::vector<int> procCoords = lattice_->get_procCoordinates();
    const int envWidth = lattice_->get_envelopeWidth();
    const int nx = lattice_->get_nx();
    const int ny = lattice_->get_ny();
    const int nz = lattice_->get_nz();
    const int ny_nz = ny * nz;
    
    // Precompute domain offsets
    double offset_x = 0.0;
    for(int iproc = 0; iproc < procCoords[0]; iproc++) {
        offset_x += lattice_->get_nxLocal(iproc);
    }
    offset_x -= (procCoords[0] * 2.0 * envWidth) + 1;
    
    double offset_y = 0.0;
    for(int jproc = 0; jproc < procCoords[1]; jproc++) {
        offset_y += lattice_->get_nyLocal(jproc);
    }
    offset_y -= (procCoords[1] * 2.0 * envWidth) + 1;
    
    double offset_z = 0.0;
    for(int kproc = 0; kproc < procCoords[2]; kproc++) {
        offset_z += lattice_->get_nzLocal(kproc);
    }
    offset_z -= (procCoords[2] * 2.0 * envWidth) + 1;
    
    // Convert particle position to lattice units
    const double x_lb_global = unitConversion->get_pos_lb(xPart[0]-origin[0]);
    const double y_lb_global = unitConversion->get_pos_lb(xPart[1]-origin[1]);
    const double z_lb_global = unitConversion->get_pos_lb(xPart[2]-origin[2]);
    
    const double x_lb_local = x_lb_global - offset_x;
    const double y_lb_local = y_lb_global - offset_y;
    const double z_lb_local = z_lb_global - offset_z;
    
    const double r_lb = unitConversion->get_radius_lb(rp);
    const int nodeZoneExtension = 5;
    
    // Calculate bounding box
    int x_min = static_cast<int>(std::floor(x_lb_local - r_lb - nodeZoneExtension));
    int x_max = static_cast<int>(std::ceil(x_lb_local + r_lb + nodeZoneExtension));
    int y_min = static_cast<int>(std::floor(y_lb_local - r_lb - nodeZoneExtension));
    int y_max = static_cast<int>(std::ceil(y_lb_local + r_lb + nodeZoneExtension));
    int z_min, z_max;
    
    if (dimension == 3) {
        z_min = static_cast<int>(std::floor(z_lb_local - r_lb - nodeZoneExtension));
        z_max = static_cast<int>(std::ceil(z_lb_local + r_lb + nodeZoneExtension));
    } else {
        z_min = 0;
        z_max = 1;
    }
    
    // Clamp to domain bounds
    x_min = std::max(envWidth, std::min(x_min, nx-envWidth));
    x_max = std::max(envWidth, std::min(x_max, nx-envWidth));
    y_min = std::max(envWidth, std::min(y_min, ny-envWidth));
    y_max = std::max(envWidth, std::min(y_max, ny-envWidth));
    z_min = std::max(envWidth, std::min(z_min, nz-envWidth));
    z_max = std::max(envWidth, std::min(z_max, nz-envWidth));
    
    // Precompute conversion factors
    const double forceFactor = unitConversion->get_forceFactor();
    const double torqueFactor = unitConversion->get_torqueFactor();
    
    // Process nodes in bounding box
    for(int i = x_min; i <= x_max; ++i) {
        const double dx = static_cast<double>(i) - x_lb_local;
        
        for(int j = y_min; j <= y_max; ++j) {
            const double dy = static_cast<double>(j) - y_lb_local;
            const int base_ij = i * ny_nz + j * nz;
            
            for(int k = z_min; k <= z_max; ++k) {
                const double dz = static_cast<double>(k) - z_lb_local;
                const int ind_phys_1D = base_ij + k;
                
                // Get particle data from lattice
                const auto& nodeData = lattice_->getParticleDataOnLatticeNode(ind_phys_1D);
                double Fhyd[3] = {0.0, 0.0, 0.0};
                
                if (tag == nodeData.particleID[0]) {
                    Fhyd[0] = nodeData.hydrodynamicForce[0];
                    Fhyd[1] = nodeData.hydrodynamicForce[1];
                    Fhyd[2] = nodeData.hydrodynamicForce[2];
                } else if (tag == nodeData.particleID[1]) {
                    Fhyd[0] = nodeData.hydrodynamicForce[3];
                    Fhyd[1] = nodeData.hydrodynamicForce[4];
                    Fhyd[2] = nodeData.hydrodynamicForce[5];
                } else {
                    continue;
                }
                
                // Convert to physical units and accumulate
                fHydro[0] += Fhyd[0] * forceFactor;
                fHydro[1] += Fhyd[1] * forceFactor;
                fHydro[2] += Fhyd[2] * forceFactor;
                
                // Calculate torque
                tHydro[0] += (dy*Fhyd[2] - dz*Fhyd[1]) * torqueFactor;
                tHydro[1] += (dz*Fhyd[0] - dx*Fhyd[2]) * torqueFactor;
                tHydro[2] += (dx*Fhyd[1] - dy*Fhyd[0]) * torqueFactor;
                
                // Calculate stresslet (symmetric part of stress tensor)
                stresslet[0] -= dx*Fhyd[0] * torqueFactor;
                stresslet[1] -= dy*Fhyd[1] * torqueFactor;
                stresslet[2] -= dz*Fhyd[2] * torqueFactor;
                stresslet[3] -= 0.5 * (dx*Fhyd[1] + dy*Fhyd[0]) * torqueFactor;
                stresslet[4] -= 0.5 * (dx*Fhyd[2] + dz*Fhyd[0]) * torqueFactor;
                stresslet[5] -= 0.5 * (dy*Fhyd[2] + dz*Fhyd[1]) * torqueFactor;
            }
        }
    }
}