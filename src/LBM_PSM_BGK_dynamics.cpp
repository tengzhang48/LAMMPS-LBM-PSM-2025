/*------------------------------------------------------ 
This file is part of the LAMMPS-LBM-PSM project.

LAMMPS-LBM-PSM is an open-source project distributed
under the GNU General Public License.

See the README and License file in the top-level 
LAMMPS-LBM-PSM directory for more details.

Tim Najuch, 2022
------------------------------------------------------*/

#include <vector>
#include "LBM_PSM_BGK_dynamics.h"

LBMPSMBGKDynamics::LBMPSMBGKDynamics(double tau_, int nx_, int ny_, int nz_, const std::vector<double>& F_lbm_, 
                                     int decomposition_[3], int procCoordinates_[3], 
                                     const std::vector<double>& origin_, const std::vector<double>& boxLength_, 
                                     int dimension_)
    : LBMPSMDynamics(nx_, ny_, nz_, decomposition_, procCoordinates_, origin_, boxLength_, dimension_),
      tau(tau_), omega(1.0/tau_), F_lbm(F_lbm_)
{
    F_lbm_mag_pow2 = F_lbm[0]*F_lbm[0] + F_lbm[1]*F_lbm[1] + F_lbm[2]*F_lbm[2];
}

LBMPSMBGKDynamics::~LBMPSMBGKDynamics() = default;

void LBMPSMBGKDynamics::initialise_dynamics(double rho_, double ux_, double uy_, double uz_) {
  for(int i = 0; i < nx; ++i) {
    for(int j = 0; j < ny; ++j) {
      for(int k = 0; k < nz; ++k) {
        const int ind_phys_1D = index_1D(i, j, k);
        const int ind_phys_2D = index_2D(i, j, k, 0);
        
        rho[ind_phys_1D] = rho_;
        u[ind_phys_2D] = ux_;
        u[ind_phys_2D+1] = uy_;
        u[ind_phys_2D+2] = uz_;

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

void LBMPSMBGKDynamics::compute_macro_values(int i_, int j_, int k_, int currentStep_) {
  const int ind_phys_1D = index_1D(i_, j_, k_);
  const int ind_phys_2D = index_2D(i_, j_, k_, 0);

  double rho_tmp = 0.0;
  double jx = 0.0;
  double jy = 0.0;
  double jz = 0.0;
  
  for(int iq = 0; iq < q; ++iq) {
    const int ind_iq = index_fi(i_, j_, k_, iq, currentStep_);
    const double f_val = f[ind_iq];
    rho_tmp += f_val;
    jx += f_val * e[3*iq];
    jy += f_val * e[3*iq+1];
    jz += f_val * e[3*iq+2];
  }
  
  rho[ind_phys_1D] = rho_tmp;

  jx += 0.5 * F_lbm[0];
  jy += 0.5 * F_lbm[1];
  jz += 0.5 * F_lbm[2];

  u[ind_phys_2D]   = jx / rho_tmp;
  u[ind_phys_2D+1] = jy / rho_tmp;
  u[ind_phys_2D+2] = jz / rho_tmp;
}

void LBMPSMBGKDynamics::collisionAndStream(int i_, int j_, int k_, int iq_, 
                                          int iShift_, int jShift_, int kShift_, 
                                          int currentStep_, int nextStep_) {
  const int ind_iq = index_fi(i_, j_, k_, iq_, currentStep_);
  const int ind_iq0 = index_fi(i_, j_, k_, iq_, 0);
  const int ind_phys_1D = index_1D(i_, j_, k_);
  const int ind_phys_2D = index_2D(i_, j_, k_, 0);

  f0[ind_iq0] = feq(iq_, ind_phys_1D, ind_phys_2D);

  // REVERTED TO ORIGINAL PARTICLE DATA ACCESS
  LAMMPS_NS::tagint pID1 = getParticleDataOnLatticeNode(ind_phys_1D).particleID[0];
  LAMMPS_NS::tagint pID2 = getParticleDataOnLatticeNode(ind_phys_1D).particleID[1];
  
  double B1 = 0.0;
  double B2 = 0.0;
  std::vector<double> uSolid1(3, 0.0);
  std::vector<double> uSolid2(3, 0.0);
  double f0_solid1 = 0.0;
  double f0_solid2 = 0.0;
  double solid_coll1 = 0.0;
  double solid_coll2 = 0.0;

  if (pID1 > 0) {
    B1 = getParticleDataOnLatticeNode(ind_phys_1D).solidFraction[0];
    uSolid1[0] = getParticleDataOnLatticeNode(ind_phys_1D).particleVelocity[0];
    uSolid1[1] = getParticleDataOnLatticeNode(ind_phys_1D).particleVelocity[1];
    uSolid1[2] = getParticleDataOnLatticeNode(ind_phys_1D).particleVelocity[2];
    f0_solid1 = feq(iq_, rho[ind_phys_1D], uSolid1);
    solid_coll1 = f0_solid1 - f[ind_iq] + (1.0 - omega) * (f[ind_iq] - f0[ind_iq0]);
  }

  if (pID2 > 0) {
    B2 = getParticleDataOnLatticeNode(ind_phys_1D).solidFraction[1];
    uSolid2[0] = getParticleDataOnLatticeNode(ind_phys_1D).particleVelocity[3];
    uSolid2[1] = getParticleDataOnLatticeNode(ind_phys_1D).particleVelocity[4];
    uSolid2[2] = getParticleDataOnLatticeNode(ind_phys_1D).particleVelocity[5];
    f0_solid2 = feq(iq_, rho[ind_phys_1D], uSolid2);
    solid_coll2 = f0_solid2 - f[ind_iq] + (1.0 - omega) * (f[ind_iq] - f0[ind_iq0]);
  }

  double Btot = B1 + B2;
  if(Btot > 1.0) {
    B1 /= Btot;
    B2 /= Btot;
    Btot = 1.0;
  }

  double F_lbm_iq = 0.0;
  if (F_lbm_mag_pow2 > 0.0) {
    F_lbm_iq = (1.0 - 0.5*omega) * F_iq(iq_, ind_phys_2D, F_lbm);
  }

  const double coll_stream = 
    f[ind_iq] 
    + (1.0 - Btot) * omega * (f0[ind_iq0] - f[ind_iq])
    + B1 * solid_coll1
    + B2 * solid_coll2
    + F_lbm_iq;
  
  set_f(iShift_, jShift_, kShift_, iq_, nextStep_, coll_stream);

  if (pID1 > 0) {
    const double force_comp = -B1 * solid_coll1;
    add_Fhyd(ind_phys_1D, pID1, force_comp * e[3*iq_], 0);
    add_Fhyd(ind_phys_1D, pID1, force_comp * e[3*iq_+1], 1);
    add_Fhyd(ind_phys_1D, pID1, force_comp * e[3*iq_+2], 2);
  }
  if (pID2 > 0) {
    const double force_comp = -B2 * solid_coll2;
    add_Fhyd(ind_phys_1D, pID2, force_comp * e[3*iq_], 0);
    add_Fhyd(ind_phys_1D, pID2, force_comp * e[3*iq_+1], 1);
    add_Fhyd(ind_phys_1D, pID2, force_comp * e[3*iq_+2], 2);
  }
}

void LBMPSMBGKDynamics::macroCollideStream() {
  for(int i = 0; i < nx; ++i) {
    for(int j = 0; j < ny; ++j) {
      for(int k = 0; k < nz; ++k) {
        compute_macro_values(i, j, k, currentStep);

        for(int iq = 0; iq < q; ++iq) {
          const int iShift = i + e[3*iq];
          const int jShift = j + e[3*iq+1];
          const int kShift = k + e[3*iq+2];

          if (iShift < 0 || iShift >= nx || 
              jShift < 0 || jShift >= ny || 
              kShift < 0 || kShift >= nz) {
            continue;
          }

          collisionAndStream(i, j, k, iq, iShift, jShift, kShift, currentStep, nextStep);
        }
      }
    }
  }

  std::swap(currentStep, nextStep);
}