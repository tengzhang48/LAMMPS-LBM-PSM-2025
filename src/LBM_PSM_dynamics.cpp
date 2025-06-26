/*------------------------------------------------------ 
This file is part of the LAMMPS-LBM-PSM project.

LAMMPS-LBM-PSM is an open-source project distributed
under the GNU General Public License.

See the README and License file in the top-level 
LAMMPS-LBM-PSM directory for more details.

Tim Najuch, 2022
------------------------------------------------------*/

#include "LBM_PSM_dynamics.h"
#include <cmath>

LBMPSMDynamics::LBMPSMDynamics(int nx_, int ny_, int nz_, int decomposition_[3], 
                               int procCoordinates_[3], const std::vector<double>& origin_, 
                               const std::vector<double>& boxLength_, int dimension_)
    : LBMPSMLattice(nx_, ny_, nz_, decomposition_, procCoordinates_, origin_, boxLength_, dimension_)
{
    for(int i = 0; i < nx; ++i) {
        for(int j = 0; j < ny; ++j) {
            for(int k = 0; k < nz; ++k) {
                const int ind_phys_1D = index_1D(i, j, k);
                const int ind_phys_2D = index_2D(i, j, k, 0);
                
                // FIXED: Calculate unique equilibrium for each direction
                for(int iq = 0; iq < q; ++iq) {
                    const double f_eq = feq(iq, ind_phys_1D, ind_phys_2D);
                    set_f0(i, j, k, iq, f_eq);
                    set_f(i, j, k, iq, 0, f_eq);
                    set_f(i, j, k, iq, 1, f_eq);
                }
            }
        }
    }
}

LBMPSMDynamics::~LBMPSMDynamics() = default;

double LBMPSMDynamics::feq(int iq_, int ind_phys_1D_, int ind_phys_2D_) const {
    const double* u_ptr = &u[ind_phys_2D_];
    const double ei_dot_u = e[3*iq_]*u_ptr[0] + e[3*iq_+1]*u_ptr[1] + e[3*iq_+2]*u_ptr[2];
    const double u_sq = u_ptr[0]*u_ptr[0] + u_ptr[1]*u_ptr[1] + u_ptr[2]*u_ptr[2];
    
    return rho[ind_phys_1D_] * w[iq_] * 
           (1.0 + ei_dot_u * invCsPow2 + 
           0.5 * ei_dot_u * ei_dot_u * invCsPow4 - 
           0.5 * u_sq * invCsPow2);
}

double LBMPSMDynamics::feq(int iq_, double rho_, const std::vector<double>& u_) const {
    const double ei_dot_u = e[3*iq_]*u_[0] + e[3*iq_+1]*u_[1] + e[3*iq_+2]*u_[2];
    const double u_sq = u_[0]*u_[0] + u_[1]*u_[1] + u_[2]*u_[2];
    
    return rho_ * w[iq_] * 
           (1.0 + ei_dot_u * invCsPow2 + 
           0.5 * ei_dot_u * ei_dot_u * invCsPow4 - 
           0.5 * u_sq * invCsPow2);
}

double LBMPSMDynamics::F_iq(int iq_, const std::vector<double>& u_, const std::vector<double>& F_) const {
    const double ei_dot_u = e[3*iq_]*u_[0] + e[3*iq_+1]*u_[1] + e[3*iq_+2]*u_[2];
    const double ei_dot_F = e[3*iq_]*F_[0] + e[3*iq_+1]*F_[1] + e[3*iq_+2]*F_[2];
    
    // FIXED: Use e - u (original sign convention)
    const double e_minus_u[3] = {e[3*iq_] - u_[0], 
                                 e[3*iq_+1] - u_[1], 
                                 e[3*iq_+2] - u_[2]};
    
    return w[iq_] * (invCsPow2 * (e_minus_u[0]*F_[0] + 
                                 e_minus_u[1]*F_[1] + 
                                 e_minus_u[2]*F_[2]) + 
                     invCsPow4 * ei_dot_u * ei_dot_F);
}

double LBMPSMDynamics::F_iq(int iq_, int ind_phys_2D_, const std::vector<double>& F_) const {
    const double* u_ptr = &u[ind_phys_2D_];
    const double ei_dot_u = e[3*iq_]*u_ptr[0] + e[3*iq_+1]*u_ptr[1] + e[3*iq_+2]*u_ptr[2];
    const double ei_dot_F = e[3*iq_]*F_[0] + e[3*iq_+1]*F_[1] + e[3*iq_+2]*F_[2];
    
    // FIXED: Use e - u (original sign convention)
    const double e_minus_u[3] = {e[3*iq_] - u_ptr[0], 
                                 e[3*iq_+1] - u_ptr[1], 
                                 e[3*iq_+2] - u_ptr[2]};
    
    return w[iq_] * (invCsPow2 * (e_minus_u[0]*F_[0] + 
                                 e_minus_u[1]*F_[1] + 
                                 e_minus_u[2]*F_[2]) + 
                     invCsPow4 * ei_dot_u * ei_dot_F);
}