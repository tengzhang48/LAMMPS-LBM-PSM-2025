/*------------------------------------------------------ 
This file is part of the LAMMPS-LBM-PSM project.

LAMMPS-LBM-PSM is an open-source project distributed
under the GNU General Public License.

See the README and License file in the top-level 
LAMMPS-LBM-PSM directory for more details.

Tim Najuch, 2022
------------------------------------------------------*/

#ifdef FIX_CLASS
// This section is for style registration
FixStyle(lbm-psm,fix_LBM_PSM)
#else
// This section is the normal class definition
#ifndef LMP_F极X_LBM_PSM_H
#define LMP_FIX_LBM_PSM_H

#include "fix.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <utility>

// Include LAMMPS core headers
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "update.h"
#include "utils.h"  // For utils::logmesg

// Include LBM-PSM headers
#include "LBM_PSM_BGK_dynamics.h"
#include "LBM_PSM_MPICOMM.h"
#include "LBM_PSM_exchangeParticleData.h"
#include "LBM_PSM_unitConversion.h"

namespace LAMMPS_NS {

class fix_LBM_PSM : public Fix {
  public:
    fix_LBM_PSM(class LAMMPS *, int, char **);
    ~fix_LBM_PSM();
    int setmask() override;
    void init() override;
    void post_force(int) override;
    int get_nevery() const { return nevery; }  // Public accessor

    double get_rho();

    class LBMPSMBGKDynamics *dynamics;
    class UnitConversion *unitConversion;
    class ExchangeParticleData *exchangeParticleData;
    class LBMPSMMPI *lbmmpicomm;

  private:
    int Nlc;
    double lc;
    double rho;
    double nu;
    double Re;
    double tau;
    std::vector<double> F_ext;
    int nevery;  // Add missing declaration

    void grow_arrays(int);
    void copy_arrays(int, int, int);
    int pack_exchange(int, double *);
    int unpack_exchange(int, double *);

    int pack_reverse_comm_size(int, int);
    int pack_reverse_comm(int, int, double *);
    void unpack_reverse_comm(int, int *, double *);

    double **hydrodynamicInteractions;
};

} // namespace LAMMPS_NS

#endif
#endif  // Add this to close the #else block