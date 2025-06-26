/*------------------------------------------------------ 
This file is part of the LAMMPS-LBM-PSM project.

LAMMPS-LBM-PSM is an open-source project distributed
under the GNU General Public License.

See the README and License file in the top-level 
LAMMPS-LBM-PSM directory for more details.

Tim Najuch, 2022
------------------------------------------------------*/

#include "LBM_PSM_zou_he_BC.h"

ZouHeBC::ZouHeBC(LBMPSMLattice *lattice_) : 
    lattice(lattice_),
    oneOverThree(1.0/3.0),
    twoOverThree(2.0/3.0),
    oneOverSix(1.0/6.0) 
{}

// 2D Velocity BC: X Negative
void ZouHeBC::setZouHeVelBC2D_xn(int ix_, int iy0_, int iy1_, double ux_bc_, int currentStep) {
    const double denom = 1.0/(1.0 - ux_bc_);
    const double uxFactor = ux_bc_ * oneOverSix;
    const double uxFactor2 = twoOverThree * ux_bc_;
    
    for (int j = iy0_; j <= iy1_; ++j) {
        const int ind_iq = lattice->index_fi(ix_, j, 0, 0, currentStep);
        
        const double f0 = lattice->get_f(ind_iq + 0);
        const double f2 = lattice->get_f(ind_iq + 2);
        const double f3 = lattice->get_f(ind_iq + 3);
        const double f4 = lattice->get_f(ind_iq + 4);
        const double f6 = lattice->get_f(ind_iq + 6);
        const double f7 = lattice->get_f(ind_iq + 7);
        
        const double rho_tmp = denom * (f0 + f2 + f4 + 2.0*(f3 + f6 + f7));
        
        lattice->set_f(ind_iq + 5, f7 + 0.5*(f4 - f2) + rho_tmp * uxFactor);
        lattice->set_f(ind_iq + 8, f6 + 0.5*(f2 - f4) + rho_tmp * uxFactor);
        lattice->set_f(ind_iq + 1, f3 + rho_tmp * uxFactor2);
    }
}

// 2D Velocity BC: X Positive (CRITICAL FIX: denominator sign)
void ZouHeBC::setZouHeVelBC2D_xp(int ix_, int iy0_, int iy1_, double ux_bc_, int currentStep) {
    const double denom = 1.0/(1.0 + ux_bc_);  // Fixed denominator sign
    const double uxFactor = ux_bc_ * oneOverSix;
    const double uxFactor2 = twoOverThree * ux_bc_;
    
    for (int j = iy0_; j <= iy1_; ++j) {
        const int ind_iq = lattice->index_fi(ix_, j, 0, 0, currentStep);
        
        const double f0 = lattice->get_f(ind_iq + 0);
        const double f1 = lattice->get_f(ind_iq + 1);
        const double f2 = lattice->get_f(ind_iq + 2);
        const double f4 = lattice->get_f(ind_iq + 4);
        const double f5 = lattice->get_f(ind_iq + 5);
        const double f8 = lattice->get_f(ind_iq + 8);
        
        const double rho_tmp = denom * (f0 + f2 + f4 + 2.0*(f1 + f5 + f8));
        
        lattice->set_f(ind_iq + 7, f5 + 0.5*(f2 - f4) - rho_tmp * uxFactor);
        lattice->set_f(ind_iq + 6, f8 + 0.5*(f4 - f2) - rho_tmp * uxFactor);
        lattice->set_f(ind_iq + 3, f1 - rho_tmp * uxFactor2);
    }
}

// 2D Density BC: X Positive
void ZouHeBC::setZouHeDensBC2D_xp(int ix_, int iy0_, int iy1_, double rho_bc_, int currentStep) {
    const double uxFactor = oneOverSix;
    
    for (int j = iy0_; j <= iy1_; ++j) {
        const int ind_iq = lattice->index_fi(ix_, j, 0, 0, currentStep);
        
        const double f0 = lattice->get_f(ind_iq + 0);
        const double f1 = lattice->get_f(ind_iq + 1);
        const double f2 = lattice->get_f(ind_iq + 2);
        const double f4 = lattice->get_f(ind_iq + 4);
        const double f5 = lattice->get_f(ind_iq + 5);
        const double f8 = lattice->get_f(ind_iq + 8);
        
        const double ux_tmp = -1.0 + 1.0/rho_bc_*(f0 + f2 + f4 + 2.0*(f1 + f5 + f8));
        
        lattice->set_f(ind_iq + 6, f8 + 0.5*(f4 - f2) - rho_bc_ * ux_tmp * uxFactor);
        lattice->set_f(ind_iq + 7, f5 + 0.5*(f2 - f4) - rho_bc_ * ux_tmp * uxFactor);
        lattice->set_f(ind_iq + 3, f1 - twoOverThree * rho_bc_ * ux_tmp);
    }
}

// 2D Velocity BC: Y Negative (Fixed argument name and signs)
void ZouHeBC::setZouHeVelBC2D_yn(int iy_, int ix0_, int ix1_, double ux_bc_, int currentStep) {
    const double uxFactor = 0.5 * ux_bc_;
    
    for (int i = ix0_; i <= ix1_; ++i) {
        const int ind_iq = lattice->index_fi(i, iy_, 0, 0, currentStep);
        
        const double f0 = lattice->get_f(ind_iq + 0);
        const double f1 = lattice->get_f(ind_iq + 1);
        const double f3 = lattice->get_f(ind_iq + 3);
        const double f4 = lattice->get_f(ind_iq + 4);
        const double f7 = lattice->get_f(ind_iq + 7);
        const double f8 = lattice->get_f(ind_iq + 8);
        
        const double rho_tmp = f0 + f1 + f3 + 2.0*(f8 + f4 + f7);
        
        lattice->set_f(ind_iq + 6, f8 + 0.5*(f1 - f3) - rho_tmp * uxFactor);
        lattice->set_f(ind_iq + 5, f7 + 0.5*(f3 - f1) + rho_tmp * uxFactor);
        lattice->set_f(ind_iq + 2, f4);
    }
}

// 2D Velocity BC: Y Positive (Fixed argument name and missing f2)
void ZouHeBC::setZouHeVelBC2D_yp(int iy_, int ix0_, int ix1_, double ux_bc_, int currentStep) {
    const double uxFactor = 0.5 * ux_bc_;
    
    for (int i = ix0_; i <= ix1_; ++i) {
        const int ind_iq = lattice->index_fi(i, iy_, 0, 0, currentStep);
        
        const double f0 = lattice->get_f(ind_iq + 0);
        const double f1 = lattice->get_f(ind_iq + 1);
        const double f2 = lattice->get_f(ind_iq + 2);  // Added missing distribution
        const double f3 = lattice->get_f(ind_iq + 3);
        const double f5 = lattice->get_f(ind_iq + 5);
        const double f6 = lattice->get_f(ind_iq + 6);
        
        const double rho_tmp = f0 + f1 + f3 + 2.0*(f6 + f2 + f5);  // Added f2
        
        lattice->set_f(ind_iq + 7, f5 + 0.5*(f1 - f3) - rho_tmp * uxFactor);
        lattice->set_f(ind_iq + 8, f6 + 0.5*(f3 - f1) + rho_tmp * uxFactor);
        lattice->set_f(ind_iq + 4, f2);
    }
}

// 2D Neumann Velocity BC: Y Negative (Fixed argument name)
void ZouHeBC::setZouHeNeumannVelBC2D_yn(int iy_, int ix0_, int ix1_, int currentStep) {
    const double factor = 0.5;
    
    for (int i = ix0_; i <= ix1_; ++i) {
        const int ind_iq = lattice->index_fi(i, iy_, 0, 0, currentStep);
        const int ind_u_neighbour_1D = lattice->index_1D(i, iy_+1, 0);
        const double u_neighbour = lattice->get_u_at_node(ind_u_neighbour_1D, 0);
        
        const double f0 = lattice->get_f(ind_iq + 0);
        const double f1 = lattice->get_f(ind_iq + 1);
        const double f3 = lattice->get_f(ind_iq + 3);
        const double f4 = lattice->get_f(ind_iq + 4);
        const double f7 = lattice->get_f(ind_iq + 7);
        const double f8 = lattice->get_f(ind_iq + 8);
        
        const double rho_tmp = f0 + f1 + f3 + 2.0*(f8 + f4 + f7);
        const double u_factor = rho_tmp * u_neighbour * factor;
        
        lattice->set_f(ind_iq + 6, f8 + 0.5*(f1 - f3) - u_factor);
        lattice->set_f(ind_iq + 5, f7 + 0.5*(f3 - f1) + u_factor);
        lattice->set_f(ind_iq + 2, f4);
    }
}

// 2D Neumann Velocity BC: Y Positive (Fixed argument name)
void ZouHeBC::setZouHeNeumannVelBC2D_yp(int iy_, int ix0_, int ix1_, int currentStep) {
    const double factor = 0.5;
    
    for (int i = ix0_; i <= ix1_; ++i) {
        const int ind_iq = lattice->index_fi(i, iy_, 0, 0, currentStep);
        const int ind_u_neighbour_1D = lattice->index_1D(i, iy_-1, 0);
        const double u_neighbour = lattice->get_u_at_node(ind_u_neighbour_1D, 0);
        
        const double f0 = lattice->get_f(ind_iq + 0);
        const double f1 = lattice->get_f(ind_iq + 1);
        const double f2 = lattice->get_f(ind_iq + 2);
        const double f3 = lattice->get_f(ind_iq + 3);
        const double f5 = lattice->get_f(ind_iq + 5);
        const double f6 = lattice->get_f(ind_iq + 6);
        
        const double rho_tmp = f0 + f1 + f3 + 2.0*(f6 + f2 + f5);
        const double u_factor = rho_tmp * u_neighbour * factor;
        
        lattice->set_f(ind_iq + 7, f5 + 0.5*(f1 - f3) - u_factor);
        lattice->set_f(ind_iq + 8, f6 + 0.5*(f3 - f1) + u_factor);
        lattice->set_f(ind_iq + 4, f2);
    }
}

// 3D Velocity BC: X Negative
void ZouHeBC::setZouHeVelBC3D_xn(int ix_, int iy0_, int iy1_, int iz0_, int iz1_, 
                                 double ux_bc_, double uy_bc_, double uz_bc_, int currentStep) {
    const double denom = 1.0/(1.0 - ux_bc_);
    
    for (int j = iy0_; j <= iy1_; ++j) {
        for (int k = iz0_; k <= iz1_; ++k) {
            const int ind_iq = lattice->index_fi(ix_, j, k, 0, currentStep);
            
            double f[19];
            for (int idx = 0; idx < 19; ++idx) {
                f[idx] = lattice->get_f(ind_iq + idx);
            }
            
            const double rho_tmp = denom * (
                f[0] + f[3] + f[4] + f[5] + f[6] + f[11] + f[17] + f[18] + f[12] +
                2.0*(f[2] + f[14] + f[8] + f[16] + f[10])
            );
            
            const double Nyx = 0.5*(f[3] + f[11] + f[17] - (f[4] + f[18] + f[12])) 
                             - oneOverThree*rho_tmp*uy_bc_;
            
            const double Nzx = 0.5*(f[5] + f[18] + f[11] - (f[6] + f[17] + f[12])) 
                             - oneOverThree*rho_tmp*uz_bc_;

            lattice->set_f(ind_iq + 1,  f[2] + oneOverThree*rho_tmp*ux_bc_);
            lattice->set_f(ind_iq + 13, f[14] + rho_tmp*oneOverSix*(ux_bc_ - uy_bc_) + Nyx);
            lattice->set_f(ind_iq + 7,  f[8]  + rho_tmp*oneOverSix*(ux_bc_ + uy_bc_) - Nyx);
            lattice->set_f(ind_iq + 9,  f[10] + rho_tmp*oneOverSix*(ux_bc_ + uz_bc_) - Nzx);
            lattice->set_f(ind_iq + 15, f[16] + rho_tmp*oneOverSix*(ux_bc_ - uz_bc_) + Nzx);
        }
    }
}

// 3D Velocity BC: X Positive
void ZouHeBC::setZouHeVelBC3D_xp(int ix_, int iy0_, int iy1_, int iz0_, int iz1_, 
                                 double ux_bc_, double uy_bc_, double uz_bc_, int currentStep) {
    const double denom = 1.0/(1.0 + ux_bc_);
    
    for (int j = iy0_; j <= iy1_; ++j) {
        for (int k = iz0_; k <= iz1_; ++k) {
            const int ind_iq = lattice->index_fi(ix_, j, k, 0, currentStep);
            
            double f[19];
            for (int idx = 0; idx < 19; ++idx) {
                f[idx] = lattice->get_f(ind_iq + idx);
            }
            
            const double rho_tmp = denom * (
                f[0] + f[3] + f[4] + f[5] + f[6] + f[11] + f[17] + f[18] + f[12] +
                2.0*(f[1] + f[7] + f[13] + f[9] + f[15])
            );
            
            const double Nyx = 0.5*(f[3] + f[11] + f[17] - (f[4] + f[18] + f[12])) 
                             - oneOverThree*rho_tmp*uy_bc_;
            
            const double Nzx = 0.5*(f[5] + f[18] + f[11] - (f[6] + f[17] + f[12])) 
                             - oneOverThree*rho_tmp*uz_bc_;

            lattice->set_f(ind_iq + 2,  f[1] - oneOverThree*rho_tmp*ux_bc_);
            lattice->set_f(ind_iq + 14, f[13] + rho_tmp*oneOverSix*(-ux_bc_ + uy_bc_) - Nyx);
            lattice->set_f(ind_iq + 8,  f[7]  + rho_tmp*oneOverSix*(-ux_bc_ - uy_bc_) + Nyx);
            lattice->set_f(ind_iq + 10, f[9]  + rho_tmp*oneOverSix*(-ux_bc_ - uz_bc_) + Nzx);
            lattice->set_f(ind_iq + 16, f[15] + rho_tmp*oneOverSix*(-ux_bc_ + uz_bc_) - Nzx);
        }
    }
}

// 3D Density BC: X Positive (Fixed term expressions)
void ZouHeBC::setZouHeDensBC3D_xp(int ix_, int iy0_, int iy1_, int iz0_, int iz1_, 
                                  double rho_bc_, double uy_bc_, double uz_bc_, int currentStep) {
    for (int j = iy0_; j <= iy1_; ++j) {
        for (int k = iz0_; k <= iz1_; ++k) {
            const int ind_iq = lattice->index_fi(ix_, j, k, 0, currentStep);
            
            double f[19];
            for (int idx = 0; idx < 19; ++idx) {
                f[idx] = lattice->get_f(ind_iq + idx);
            }
            
            const double ux_bc_ = -1.0 + 1.0/rho_bc_*(
                f[0] + f[3] + f[4] + f[5] + f[6] + f[11] + f[17] + f[18] + f[12] +
                2.0*(f[1] + f[7] + f[13] + f[9] + f[15])
            );
            
            const double Nyx = 0.5*(f[3] + f[11] + f[17] - (f[4] + f[18] + f[12])) 
                             - oneOverThree*rho_bc_*uy_bc_;
            
            const double Nzx = 0.5*(f[5] + f[18] + f[11] - (f[6] + f[17] + f[12])) 
                             - oneOverThree*rho_bc_*uz_bc_;

            lattice->set_f(ind_iq + 2,  f[1] - oneOverThree*rho_bc_*ux_bc_);
            lattice->set_f(ind_iq + 14, f[13] + rho_bc_*oneOverSix*(-ux_bc_ + uy_bc_) - Nyx);
            lattice->set_f(ind_iq + 8,  f[7]  + rho_bc_*oneOverSix*(-ux_bc_ - uy_bc_) + Nyx);
            lattice->set_f(ind_iq + 10, f[9]  + rho_bc_*oneOverSix*(-ux_bc_ - uz_bc_) + Nzx);
            lattice->set_f(ind_iq + 16, f[15] + rho_bc_*oneOverSix*(-ux_bc_ + uz_bc_) - Nzx);
        }
    }
}

// 3D Velocity BC: Y Negative
void ZouHeBC::setZouHeVelBC3D_yn(int iy_, int ix0_, int ix1_, int iz0_, int iz1_, 
                                 double ux_bc_, double uy_bc_, double uz_bc_, int currentStep) {
    const double denom = 1.0/(1.0 - uy_bc_);
    
    for (int i = ix0_; i <= ix1_; ++i) {
        for (int k = iz0_; k <= iz1_; ++k) {
            const int ind_iq = lattice->index_fi(i, iy_, k, 0, currentStep);
            
            double f[19];
            for (int idx = 0; idx < 19; ++idx) {
                f[idx] = lattice->get_f(ind_iq + idx);
            }
            
            const double rho_tmp = denom * (
                f[0] + f[1] + f[2] + f[5] + f[6] + f[9] + f[15] + f[16] + f[10] +
                2.0*(f[4] + f[13] + f[8] + f[18] + f[12])
            );
            
            const double Nxy = 0.5*(f[1] + f[9] + f[15] - (f[2] + f[16] + f[10])) 
                             - oneOverThree*rho_tmp*ux_bc_;
            
            const double Nzy = 0.5*(f[5] + f[9] + f[16] - (f[6] + f[15] + f[10])) 
                             - oneOverThree*rho_tmp*uz_bc_;

            lattice->set_f(ind_iq + 3,  f[4] + oneOverThree*rho_tmp*uy_bc_);
            lattice->set_f(ind_iq + 7,  f[8]  + rho_tmp*oneOverSix*(uy_bc_ + ux_bc_) - Nxy);
            lattice->set_f(ind_iq + 14, f[13] + rho_tmp*oneOverSix*(uy_bc_ - ux_bc_) + Nxy);
            lattice->set_f(ind_iq + 11, f[12] + rho_tmp*oneOverSix*(uy_bc_ + uz_bc_) - Nzy);
            lattice->set_f(ind_iq + 17, f[18] + rho_tmp*oneOverSix*(uy_bc_ - uz_bc_) + Nzy);
        }
    }
}

// 3D Velocity BC: Y Positive
void ZouHeBC::setZouHeVelBC3D_yp(int iy_, int ix0_, int ix1_, int iz0_, int iz1_, 
                                 double ux_bc_, double uy_bc_, double uz_bc_, int currentStep) {
    const double denom = 1.0/(1.0 + uy_bc_);
    
    for (int i = ix0_; i <= ix1_; ++i) {
        for (int k = iz0_; k <= iz1_; ++k) {
            const int ind_iq = lattice->index_fi(i, iy_, k, 0, currentStep);
            
            double f[19];
            for (int idx = 0; idx < 19; ++idx) {
                f[idx] = lattice->get_f(ind_iq + idx);
            }
            
            const double rho_tmp = denom * (
                f[0] + f[1] + f[2] + f[5] + f[6] + f[9] + f[15] + f[16] + f[10] +
                2.0*(f[3] + f[7] + f[14] + f[11] + f[17])
            );
            
            const double Nxy = 0.5*(f[1] + f[9] + f[15] - (f[2] + f[16] + f[10])) 
                             - oneOverThree*rho_tmp*ux_bc_;
            
            const double Nzy = 0.5*(f[5] + f[9] + f[16] - (f[6] + f[15] + f[10])) 
                             - oneOverThree*rho_tmp*uz_bc_;

            lattice->set_f(ind_iq + 4,  f[3] - oneOverThree*rho_tmp*uy_bc_);
            lattice->set_f(ind_iq + 8,  f[7]  + rho_tmp*oneOverSix*(-uy_bc_ - ux_bc_) + Nxy);
            lattice->set_f(ind_iq + 13, f[14] + rho_tmp*oneOverSix*(-uy_bc_ + ux_bc_) - Nxy);
            lattice->set_f(ind_iq + 12, f[11] + rho_tmp*oneOverSix*(-uy_bc_ - uz_bc_) + Nzy);
            lattice->set_f(ind_iq + 18, f[17] + rho_tmp*oneOverSix*(-uy_bc_ + uz_bc_) - Nzy);
        }
    }
}

// 3D Velocity BC: Z Negative
void ZouHeBC::setZouHeVelBC3D_zn(int iz_, int ix0_, int ix1_, int iy0_, int iy1_, 
                                 double ux_bc_, double uy_bc_, double uz_bc_, int currentStep) {
    const double denom = 1.0/(1.0 - uz_bc_);
    
    for (int i = ix0_; i <= ix1_; ++i) {
        for (int j = iy0_; j <= iy1_; ++j) {
            const int ind_iq = lattice->index_fi(i, j, iz_, 0, currentStep);
            
            double f[19];
            for (int idx = 0; idx < 19; ++idx) {
                f[idx] = lattice->get_f(ind_iq + idx);
            }
            
            const double rho_tmp = denom * (
                f[0] + f[1] + f[2] + f[3] + f[4] + f[7] + f[13] + f[14] + f[8] +
                2.0*(f[6] + f[15] + f[10] + f[17] + f[12])
            );
            
            const double Nxz = 0.5*(f[1] + f[7] + f[13] - (f[2] + f[14] + f[8])) 
                             - oneOverThree*rho_tmp*ux_bc_;
            
            const double Nyz = 0.5*(f[3] + f[7] + f[14] - (f[4] + f[13] + f[8])) 
                             - oneOverThree*rho_tmp*uy_bc_;

            lattice->set_f(ind_iq + 5,  f[6] + oneOverThree*rho_tmp*uz_bc_);
            lattice->set_f(ind_iq + 9,  f[10] + rho_tmp*oneOverSix*(uz_bc_ + ux_bc_) - Nxz);
            lattice->set_f(ind_iq + 16, f[15] + rho_tmp*oneOverSix*(uz_bc_ - ux_bc_) + Nxz);
            lattice->set_f(ind_iq + 11, f[12] + rho_tmp*oneOverSix*(uz_bc_ + uy_bc_) - Nyz);
            lattice->set_f(ind_iq + 18, f[17] + rho_tmp*oneOverSix*(uz_bc_ - uy_bc_) + Nyz);
        }
    }
}

// 3D Velocity BC: Z Positive
void ZouHeBC::setZouHeVelBC3D_zp(int iz_, int ix0_, int ix1_, int iy0_, int iy1_, 
                                 double ux_bc_, double uy_bc_, double uz_bc_, int currentStep) {
    const double denom = 1.0/(1.0 + uz_bc_);
    
    for (int i = ix0_; i <= ix1_; ++i) {
        for (int j = iy0_; j <= iy1_; ++j) {
            const int ind_iq = lattice->index_fi(i, j, iz_, 0, currentStep);
            
            double f[19];
            for (int idx = 0; idx < 19; ++idx) {
                f[idx] = lattice->get_f(ind_iq + idx);
            }
            
            const double rho_tmp = denom * (
                f[0] + f[1] + f[2] + f[3] + f[4] + f[7] + f[14] + f[8] + f[13] +
                2.0*(f[5] + f[9] + f[16] + f[11] + f[18])
            );
            
            const double Nxz = 0.5*(f[1] + f[7] + f[13] - (f[2] + f[14] + f[8])) 
                             - oneOverThree*rho_tmp*ux_bc_;
            
            const double Nyz = 0.5*(f[3] + f[7] + f[14] - (f[4] + f[13] + f[8])) 
                             - oneOverThree*rho_tmp*uy_bc_;

            lattice->set_f(ind_iq + 6,  f[5] - oneOverThree*rho_tmp*uz_bc_);
            lattice->set_f(ind_iq + 15, f[16] + rho_tmp*oneOverSix*(-uz_bc_ + ux_bc_) - Nxz);
            lattice->set_f(ind_iq + 10, f[9]  + rho_tmp*oneOverSix*(-uz_bc_ - ux_bc_) + Nxz);
            lattice->set_f(ind_iq + 17, f[18] + rho_tmp*oneOverSix*(-uz_bc_ + uy_bc_) - Nyz);
            lattice->set_f(ind_iq + 12, f[11] + rho_tmp*oneOverSix*(-uz_bc_ - uy_bc_) + Nyz);
        }
    }
}