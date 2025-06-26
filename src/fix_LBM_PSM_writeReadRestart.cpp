/*------------------------------------------------------ 
This file is part of the LAMMPS-LBM-PSM project.

LAMMPS-LBM-PSM is an open-source project distributed
under the GNU General Public License.

See the README and License file in the top-level 
LAMMPS-LBM-PSM directory for more details.

Tim Najuch, 2022
------------------------------------------------------*/

#include "fix_LBM_PSM_writeReadRestart.h"
#include "fix_LBM_PSM.h"  // Now includes full definition of LBMPSMBGKDynamics
#include "error.h"
#include "modify.h"
#include "comm.h"
#include "update.h"
#include <sstream>
#include <fstream>

using namespace LAMMPS_NS;

LBMPSMWriteReadRestart::LBMPSMWriteReadRestart(LAMMPS *lmp, int narg, char **arg) : 
    Fix(lmp, narg, arg), iWrite(0), iRead(0), fixLBMPSM(nullptr) {

  if (narg < 9) error->all(FLERR,"Illegal fix lbm-psm-restart command");

  for(int ifix=0; ifix<modify->nfix; ifix++) {
    if(strcmp(modify->fix[ifix]->style,"lbm-psm") == 0) {
      fixLBMPSM = dynamic_cast<fix_LBM_PSM *>(modify->fix[ifix]);
    }
  }

  if (!fixLBMPSM) {
    error->all(FLERR,"fix lbm-psm-restart requires fix lbm-psm");
  }

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"every") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix lbm-psm-restart command");
      nevery = atoi(arg[iarg+1]);
      if (nevery <= 0) error->all(FLERR,"Illegal fix lbm-psm-restart command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"write") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix lbm-psm-restart command");
      iWrite = atoi(arg[iarg+1]);
      if (iWrite < 0 || iWrite > 1) error->all(FLERR,"Illegal fix lbm-psm-restart command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"read") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix lbm-psm-restart command");
      iRead = atoi(arg[iarg+1]);
      if (iRead < 0 || iRead > 1) error->all(FLERR,"Illegal fix lbm-psm-restart command");
      iarg += 2;
    } else error->all(FLERR,"Illegal fix lbm-psm-restart command");
  }
}

LBMPSMWriteReadRestart::~LBMPSMWriteReadRestart() {}

int LBMPSMWriteReadRestart::setmask() {
  int mask = 0;
  mask |= FixConst::PRE_FORCE;
  return mask;
}

void LBMPSMWriteReadRestart::init() {
  if (iRead == 1) {
    std::ostringstream restartFileStringTmp;
    restartFileStringTmp << "LBM-PSM-restartFile-processor-" << comm->me << ".bin";
    read_restart(restartFileStringTmp.str(), fixLBMPSM->dynamics->getVector_f());
  }
}

void LBMPSMWriteReadRestart::pre_force(int) {
  if (update->ntimestep % nevery) return;
  if (iWrite == 1) {
    std::ostringstream restartFileStringTmp;
    restartFileStringTmp << "LBM-PSM-restartFile-processor-" << comm->me << ".bin";
    write_restart(restartFileStringTmp.str(), 
                 fixLBMPSM->dynamics->getVector_f(), 
                 fixLBMPSM->dynamics->get_currentStep());
  }
}

void LBMPSMWriteReadRestart::write_restart(const std::string& name_, 
                                          std::vector<double> &f_, 
                                          int currentStep) {
  std::ofstream out(name_, std::ios::binary);
  if (!out) {
    std::string err = "Cannot open restart file " + name_ + " for writing";
    error->one(FLERR, err.c_str());
  }

  out.write(reinterpret_cast<const char*>(&currentStep), sizeof(int));
  const size_t f_size = f_.size();
  out.write(reinterpret_cast<const char*>(&f_size), sizeof(size_t));
  out.write(reinterpret_cast<const char*>(f_.data()), f_size * sizeof(double));
}

void LBMPSMWriteReadRestart::read_restart(const std::string& name_, 
                                         std::vector<double> &f_) {
  std::ifstream in(name_, std::ios::binary);
  if (!in) {
    std::string err = "Cannot open restart file " + name_ + " for reading";
    error->one(FLERR, err.c_str());
  }

  int currentStep = 0;
  in.read(reinterpret_cast<char*>(&currentStep), sizeof(int));
  fixLBMPSM->dynamics->set_currentStep(currentStep);

  size_t vsize = 0;
  in.read(reinterpret_cast<char*>(&vsize), sizeof(size_t));
  
  if (vsize != f_.size()) {
    std::ostringstream err;
    err << "Restart file size mismatch: expected " << f_.size() << ", got " << vsize;
    error->one(FLERR, err.str().c_str());
  }

  in.read(reinterpret_cast<char*>(f_.data()), vsize * sizeof(double));
}