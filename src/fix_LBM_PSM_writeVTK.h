/*------------------------------------------------------ 
This file is part of the LAMMPS-LBM-PSM project.

LAMMPS-LBM-PSM is an open-source project distributed
under the GNU General Public License.

See the README and License file in the top-level 
LAMMPS-LBM-PSM directory for more details.

Tim Najuch, 2022
------------------------------------------------------*/

#ifdef FIX_CLASS

FixStyle(lbm-psm-vtk,WriteVTK)

#else

#ifndef WRITEVTK_H
#define WRITEVTK_H

#include <vector>
#include <string>
#include "fix.h"

namespace LAMMPS_NS {

// Forward declaration
class fix_LBM_PSM;

class WriteVTK : public Fix {
  private:
    int nx, ny, nz;
    int decomposition[3];

  public:
    WriteVTK(class LAMMPS *, int, char **);
    ~WriteVTK();
    int setmask() override;
    void init() override;
    void pre_force(int) override;

    void write_vtk(std::string name_, 
                  std::vector<double> &x_, double x0_,
                  std::vector<double> &y_, double y0_,
                  std::vector<double> &z_, double z0_,
                  std::vector<double> &B_, double B0_,
                  std::vector<double> &rho_, double rho0_,
                  std::vector<double> &u_, double u0_);

    void scale_vector(std::vector<double> &vec_, double scaling_);

    class fix_LBM_PSM *fixLBMPSM;
};

} // namespace LAMMPS_NS

#endif
#endif