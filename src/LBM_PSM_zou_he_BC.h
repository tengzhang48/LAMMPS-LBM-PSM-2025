/*------------------------------------------------------ 
This file is part of the LAMMPS-LBM-PSM project.

LAMMPS-LBM-PSM is an open-source project distributed
under the GNU General Public License.

See the README and License file in the top-level 
LAMMPS-LBM-PSM directory for more details.

Tim Najuch, 2022
------------------------------------------------------*/

#ifndef ZOU_HE_BC_H
#define ZOU_HE_BC_H

#include "LBM_PSM_lattice.h"

class ZouHeBC {
private:
    LBMPSMLattice *lattice;
    const double oneOverThree;
    const double twoOverThree;
    const double oneOverSix;
    
public:
    ZouHeBC(LBMPSMLattice *lattice_);
    ~ZouHeBC() = default;

    // 2D boundary conditions
    void setZouHeVelBC2D_xn(int ix_, int iy0_, int iy1_, double ux_bc_, int currentStep);
    void setZouHeVelBC2D_xp(int ix_, int iy0_, int iy1_, double ux_bc_, int currentStep);
    void setZouHeDensBC2D_xp(int ix_, int iy0_, int iy1_, double rho_bc_, int currentStep);
    void setZouHeVelBC2D_yn(int iy_, int ix0_, int ix1_, double uy_bc_, int currentStep);
    void setZouHeVelBC2D_yp(int iy_, int ix0_, int ix1_, double uy_bc_, int currentStep);
    void setZouHeNeumannVelBC2D_yn(int iy_, int ix0_, int ix1_, int currentStep);
    void setZouHeNeumannVelBC2D_yp(int iy_, int ix0_, int ix1_, int currentStep);

    // 3D boundary conditions
    void setZouHeVelBC3D_xn(int ix_, int iy0_, int iy1_, int iz0_, int iz1_, 
                            double ux_bc_, double uy_bc_, double uz_bc_, int currentStep);
    void setZouHeVelBC3D_xp(int ix_, int iy0_, int iy1_, int iz0_, int iz1_, 
                            double ux_bc_, double uy_bc_, double uz_bc_, int currentStep);
    void setZouHeDensBC3D_xp(int ix_, int iy0_, int iy1_, int iz0_, int iz1_, 
                             double rho_bc_, double uy_bc_, double uz_bc_, int currentStep);
    void setZouHeVelBC3D_yn(int iy_, int ix0_, int ix1_, int iz0_, int iz1_, 
                            double ux_bc_, double uy_bc_, double uz_bc_, int currentStep);
    void setZouHeVelBC3D_yp(int iy_, int ix0_, int ix1_, int iz0_, int iz1_, 
                            double ux_bc_, double uy_bc_, double uz_bc_, int currentStep);
    void setZouHeVelBC3D_zn(int iz_, int ix0_, int ix1_, int iy0_, int iy1_, 
                            double ux_bc_, double uy_bc_, double uz_bc_, int currentStep);
    void setZouHeVelBC3D_zp(int iz_, int ix0_, int ix1_, int iy0_, int iy1_, 
                            double ux_bc_, double uy_bc_, double uz_bc_, int currentStep);
};

#endif