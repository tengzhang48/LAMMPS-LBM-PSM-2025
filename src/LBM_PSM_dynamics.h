/*------------------------------------------------------ 
This file is part of the LAMMPS-LBM-PSM project.

LAMMPS-LBM-PSM is an open-source project distributed
under the GNU General Public License.

See the README and License file in the top-level 
LAMMPS-LBM-PSM directory for more details.

Tim Najuch, 2022
------------------------------------------------------*/

#ifndef LBMPSMDYNAMICS_2D
#define LBMPSMDYNAMICS_2D

#include <vector>
#include "LBM_PSM_lattice.h"

class LBMPSMDynamics : public LBMPSMLattice {
public:
    LBMPSMDynamics(int nx_, int ny_, int nz_, int decomposition_[3], 
                   int procCoordinates_[3], const std::vector<double>& origin_, 
                   const std::vector<double>& boxLength_, int dimension_);
    ~LBMPSMDynamics();

    // Precompute reusable values for equilibrium calculations
    double feq(int iq_, int ind_phys_1D_, int ind_phys_2D_) const;
    double feq(int iq_, double rho_, const std::vector<double>& u_) const;

    // Optimized force term calculations
    double F_iq(int iq_, const std::vector<double>& u_, const std::vector<double>& F_) const;
    double F_iq(int iq_, int ind_phys_2D_, const std::vector<double>& F_) const;

private:
    // Precomputed constants
    static constexpr double invCsPow2 = 3.0;  // 1/c_s^2
    static constexpr double invCsPow4 = 9.0;  // 1/c_s^4
};

#endif