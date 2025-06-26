/*------------------------------------------------------ 
This file is part of the LAMMPS-LBM-PSM project.

LAMMPS-LBM-PSM is an open-source project distributed
under the GNU General Public License.

See the README and License file in the top-level 
LAMMPS-LBM-PSM directory for more details.

Tim Najuch, 2022
------------------------------------------------------*/

#include "LBM_PSM_unitConversion.h"
#include <cmath>

// Precompute all values in one place
void UnitConversion::precompute_values() {
    cs = 1.0/sqrt(3.0);
    csPow2 = cs * cs;
    nu_lb = (tau - 0.5) * csPow2;
    
    dx = lc / static_cast<double>(N - 1);
    dx_d = 1.0 / static_cast<double>(N - 1);  // Since Lc_d = 1.0
    
    Uc = Re * nu / lc;
    u_lb = Re * nu_lb / static_cast<double>(N - 1);
    dt_d = dx_d * u_lb;  // Uc_d = 1.0
    
    tc = lc / Uc;
    
    // Precompute powers and dt_d^2
    const double dx_d2 = dx_d * dx_d;
    const double dx_d4 = dx_d2 * dx_d2;
    const double dx_d5 = dx_d4 * dx_d;
    const double dt_d2 = dt_d * dt_d;
    const double tc2 = tc * tc;
    
    // CORRECTED: Added division by dt_d^2
    forceFactor = rhof * (lc*lc*lc*lc) / tc2 * dx_d4 / dt_d2;
    torqueFactor = rhof * (lc*lc*lc*lc*lc) / tc2 * dx_d5 / dt_d2;
    volumeForceFactor = rhof * lc / tc2 * dx_d / dt_d2;
}

UnitConversion::UnitConversion(double rhof_, double nu_, double lc_, double Re_, int N_, double tau_, int dimension_) : 
    rhof(rhof_), nu(nu_), lc(lc_), Re(Re_), N(N_), tau(tau_), dimension(dimension_) 
{
    precompute_values();
}

std::vector<double> UnitConversion::get_volume_force_lb(const std::vector<double>& F_phys) const {
    const double invFactor = 1.0 / volumeForceFactor;
    return {
        F_phys[0] * invFactor,
        F_phys[1] * invFactor,
        F_phys[2] * invFactor
    };
}