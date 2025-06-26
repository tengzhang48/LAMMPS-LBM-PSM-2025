/*------------------------------------------------------ 
This file is part of the LAMMPS-LBM-PSM project.

LAMMPS-LBM-PSM is an open-source project distributed
under the GNU General Public License.

See the README and License file in the top-level 
LAMMPS-LBM-PSM directory for more details.

Tim Najuch, 2022
------------------------------------------------------*/

#ifndef LBMPSM_PARTDATALATTICE_H
#define LBMPSM_PARTDATALATTICE_H

#include <array>
#include <vector>
#include "lmptype.h"

class ParticleDataOnLattice {
public:
    // Constants for fixed-size data
    static constexpr int MAX_PARTICLES_PER_NODE = 2;
    static constexpr int DIM = 3;
    
    ParticleDataOnLattice();
    ~ParticleDataOnLattice();  // Destructor declaration
    
    // Data storage using fixed-size arrays
    std::array<LAMMPS_NS::tagint, MAX_PARTICLES_PER_NODE> particleID;
    std::array<double, MAX_PARTICLES_PER_NODE> solidFraction;
    std::array<double, MAX_PARTICLES_PER_NODE * DIM> particleVelocity;
    std::array<double, MAX_PARTICLES_PER_NODE * DIM> hydrodynamicForce;
    
    // Conversion function to vector for compatibility
    std::vector<double> particleVelocityAsVector() const {
        return std::vector<double>(particleVelocity.begin(), particleVelocity.end());
    }
};

#endif