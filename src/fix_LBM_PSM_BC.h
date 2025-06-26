/*------------------------------------------------------ 
This file is part of the LAMMPS-LBM-PSM project.

LAMMPS-LBM-PSM is an open-source project distributed
under the GNU General Public License.

See the README and License file in the top-level 
LAMMPS-LBM-PSM directory for more details.

Tim Najuch, 2022
------------------------------------------------------*/

#ifdef FIX_CLASS

FixStyle(lbm-psm-bc,fix_LBM_PSM_BC)

#else

#ifndef LMP_FIX_LBM_PSM_BC_H
#define LMP_FIX_LBM_PSM_BC_H

#include "fix.h"
#include "LBM_PSM_zou_he_BC.h"  // Include full definition of ZouHeBC

namespace LAMMPS_NS {

class fix_LBM_PSM_BC : public Fix {
  public:
    fix_LBM_PSM_BC(class LAMMPS *, int, char **);
    ~fix_LBM_PSM_BC();
    int setmask() override;
    void init() override;
    void post_force(int) override;

  private:
    int typeBC;
    class fix_LBM_PSM *fixLBMPSM;
    class ZouHeBC *zouHe;
};

} // namespace LAMMPS_NS

#endif
#endif