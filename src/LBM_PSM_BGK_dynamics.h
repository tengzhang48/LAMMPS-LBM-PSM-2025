/*------------------------------------------------------ 
This file is part of the LAMMPS-LBM-PSM project.

LAMMPS-LBM-PSM is an open-source project distributed
under the GNU General Public License.

See the README and License file in the top-level 
LAMMPS-LBM-PSM directory for more details. 

Tim Najuch, 2022
------------------------------------------------------*/

#ifndef LBMPSM_BGK_DYNAMICS
#define LBMPSM_BGK_DYNAMICS

#include <vector>
#include "LBM_PSM_lattice.h"
#include "LBM_PSM_dynamics.h"
#include "domain.h"

class LBMPSMBGKDynamics : public LBMPSMDynamics {
  
  private:
    double tau, omega;      // Single relaxation parameter tau and its inverse omega
    std::vector<double> F_lbm;
    double F_lbm_mag_pow2;

  public:
    LBMPSMBGKDynamics(double tau_, int nx_, int ny_, int nz_, const std::vector<double>& F_lbm_, 
                      int decomposition_[3], int procCoordinates_[3], 
                      const std::vector<double>& origin_, const std::vector<double>& boxLength_, 
                      int dimension_);
    ~LBMPSMBGKDynamics();

    void initialise_dynamics(double rho_, double ux_, double uy_, double uz_);
    void compute_macro_values(int i_, int j_, int k_, int currentStep_);
    void collisionAndStream(int i_, int j_, int k_, int iq_, int iShift_, int jShift_, int kShift_, int currentStep_, int nextStep_);
    void macroCollideStream();
    
};

#endif