/*------------------------------------------------------ 
This file is part of the LAMMPS-LBM-PSM project.

LAMMPS-LBM-PSM is an open-source project distributed
under the GNU General Public License.

See the README and License file in the top-level 
LAMMPS-LBM-PSM directory for more details.

Tim Najuch, 2022
------------------------------------------------------*/

#ifdef FIX_CLASS

FixStyle(lbm-psm-restart,LBMPSMWriteReadRestart)

#else

#ifndef LBMPSMWRITEREADRESTART_H
#define LBMPSMWRITEREADRESTART_H

#include <vector>
#include <string>
#include "fix.h"

namespace LAMMPS_NS {

class fix_LBM_PSM;

class LBMPSMWriteReadRestart : public Fix {
  private:
    int iWrite, iRead;

  public:
    LBMPSMWriteReadRestart(class LAMMPS *, int, char **);
    ~LBMPSMWriteReadRestart();
    int setmask() override;
    void init() override;
    void pre_force(int) override;

    void write_restart(const std::string& name_, std::vector<double> &f_, int currentStep);
    void read_restart(const std::string& name_, std::vector<double> &f_);

    class fix_LBM_PSM *fixLBMPSM;
};

} // namespace LAMMPS_NS

#endif
#endif