/*------------------------------------------------------ 
This file is part of the LAMMPS-LBM-PSM project.

LAMMPS-LBM-PSM is an open-source project distributed
under the GNU General Public License.

See the README and License file in the top-level 
LAMMPS-LBM-PSM directory for more details.

Tim Najuch, 2022
------------------------------------------------------*/

#ifndef LBM_PSM_UNIT_CONVERSION_H
#define LBM_PSM_UNIT_CONVERSION_H

#include <vector>

class UnitConversion {
private:
    // Input parameters
    double rhof;
    double nu;
    double lc;
    double Re;
    int N;
    double tau;
    int dimension;

    // Precomputed constants
    double cs;
    double csPow2;
    double nu_lb;
    double dx;
    double dx_d;
    double Uc;
    double u_lb;
    double dt_d;
    double tc;    
    double forceFactor;
    double torqueFactor;
    double volumeForceFactor;

    // Helper for precomputation
    void precompute_values();

public:
    UnitConversion(double rhof_, double nu_, double lc_, double Re_, int N_, double tau_, int dimension_);
    ~UnitConversion() = default;

    // Const getter methods
    double get_dx() const { return dx; }
    double get_Uc() const { return Uc; }
    double get_radius_lb(double rp) const { return rp/dx; }
    double get_u_lb() const { return u_lb; }
    double get_vel_lb(double vel_phys) const { return vel_phys / Uc * u_lb; }
    double get_freq_lb(double freq_phys) const { return freq_phys * tc * dt_d; }
    double get_pos_lb(double pos_phys) const { return pos_phys/dx; }
    double get_forceFactor() const { return forceFactor; }
    double get_torqueFactor() const { return torqueFactor; }
    double get_volumeForceFactor() const { return volumeForceFactor; }
    double get_phys_time(double time_lb) const { return time_lb * dt_d * tc; }
    double get_dt_d() const { return dt_d; }
    
    // Vector conversion
    std::vector<double> get_volume_force_lb(const std::vector<double>& F_phys) const;
};

#endif