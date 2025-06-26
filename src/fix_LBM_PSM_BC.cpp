/*------------------------------------------------------ 
This file is part of the LAMMPS-LBM-PSM project.

LAMMPS-LBM-PSM is an open-source project distributed
under the GNU General Public License.

See the README and License file in the top-level 
LAMMPS-LBM-PSM directory for more details.

Tim Najuch, 2022
------------------------------------------------------*/

#include "fix_LBM_PSM_BC.h"
#include "fix_LBM_PSM.h"
#include "LBM_PSM_zou_he_BC.h"
#include "error.h"
#include "modify.h"
#include "domain.h"
#include "comm.h"
#include "update.h"

using namespace LAMMPS_NS;

fix_LBM_PSM_BC::fix_LBM_PSM_BC(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), typeBC(0), fixLBMPSM(nullptr), zouHe(nullptr)
{
  if (narg < 4) error->all(FLERR,"Illegal fix lbm-psm-bc command");
  
  // Parse boundary condition type
  int iarg = 3;
  if (strcmp(arg[iarg],"shear") == 0) {
    typeBC = 1;
  } else if (strcmp(arg[iarg],"xFlow") == 0) {
    typeBC = 2;
  } else if (strcmp(arg[iarg],"channel") == 0) {
    typeBC = 3;
  } else if (strcmp(arg[iarg],"channel-velIn-pressOut") == 0) {
    typeBC = 4;
  } else if (strcmp(arg[iarg],"closedBox") == 0) {
    typeBC = 5;
  } else {
    error->all(FLERR,"Illegal fix lbm-psm-bc command: unknown boundary type");
  }

  // Find associated LBM-PSM fix
  for(int ifix = 0; ifix < modify->nfix; ifix++) {
    if(strcmp(modify->fix[ifix]->style, "lbm-psm") == 0) {
      fixLBMPSM = dynamic_cast<fix_LBM_PSM *>(modify->fix[ifix]);
      break;
    }
  }
  
  if (!fixLBMPSM) {
    error->all(FLERR,"fix lbm-psm-bc requires fix lbm-psm");
  }
}

fix_LBM_PSM_BC::~fix_LBM_PSM_BC() {
  delete zouHe;
}

int fix_LBM_PSM_BC::setmask() {
  return FixConst::POST_FORCE;
}

void fix_LBM_PSM_BC::init() {
  zouHe = new ZouHeBC(fixLBMPSM->dynamics);
}

void fix_LBM_PSM_BC::post_force(int) {
  // Use public accessor for nevery
  if (update->ntimestep % fixLBMPSM->get_nevery()) return;

  const int envWidth = fixLBMPSM->dynamics->get_envelopeWidth();
  const double u_infty = fixLBMPSM->unitConversion->get_u_lb();
  const double rho_outlet = 1.0;
  const int currentStep = fixLBMPSM->dynamics->get_currentStep();
  
  // Precompute dimensions
  const int nx = fixLBMPSM->dynamics->get_nx();
  const int ny = fixLBMPSM->dynamics->get_ny();
  const int nz = fixLBMPSM->dynamics->get_nz();
  
  // Boundary condition application based on type
  if (domain->dimension == 2) {
    switch (typeBC) {
      case 1: // Shear flow
        if (comm->myloc[1] == 0) {
          zouHe->setZouHeVelBC2D_yn(envWidth, envWidth, nx-1-envWidth, -u_infty, currentStep);
        }
        if (comm->myloc[1] == comm->procgrid[1]-1) {
          zouHe->setZouHeVelBC2D_yp(ny-1-envWidth, envWidth, nx-1-envWidth, u_infty, currentStep);
        }
        break;
        
      case 2: // X-direction flow
        if (comm->myloc[0] == 0) {
          zouHe->setZouHeVelBC2D_xn(envWidth, envWidth, ny-1-envWidth, u_infty, currentStep);
        }
        if (comm->myloc[0] == comm->procgrid[0]-1) {
          zouHe->setZouHeDensBC2D_xp(nx-1-envWidth, envWidth, ny-1-envWidth, rho_outlet, currentStep);
        }
        if (comm->myloc[1] == 0) {
          zouHe->setZouHeVelBC2D_yn(envWidth, 1+envWidth, nx-2-envWidth, u_infty, currentStep);
        }
        if (comm->myloc[1] == comm->procgrid[1]-1) {
          zouHe->setZouHeVelBC2D_yp(ny-1-envWidth, 1+envWidth, nx-2-envWidth, u_infty, currentStep);
        }
        break;
        
      case 3: // Channel flow
        if (comm->myloc[1] == 0) {
          zouHe->setZouHeVelBC2D_yn(envWidth, envWidth, nx-1-envWidth, 0.0, currentStep);
        }
        if (comm->myloc[1] == comm->procgrid[1]-1) {
          zouHe->setZouHeVelBC2D_yp(ny-1-envWidth, envWidth, nx-1-envWidth, 0.0, currentStep);
        }
        break;
        
      case 4: // Channel with inlet/outlet
        if (comm->myloc[0] == 0) {
          zouHe->setZouHeVelBC2D_xn(envWidth, envWidth, ny-1-envWidth, u_infty, currentStep);
        }
        if (comm->myloc[0] == comm->procgrid[0]-1) {
          zouHe->setZouHeDensBC2D_xp(nx-1-envWidth, envWidth, ny-1-envWidth, rho_outlet, currentStep);
        }
        if (comm->myloc[1] == 0) {
          zouHe->setZouHeVelBC2D_yn(envWidth, envWidth, nx-1-envWidth, 0.0, currentStep);
        }
        if (comm->myloc[1] == comm->procgrid[1]-1) {
          zouHe->setZouHeVelBC2D_yp(ny-1-envWidth, envWidth, nx-1-envWidth, 0.0, currentStep);
        }
        break;
        
      case 5: // Closed box
        if (comm->myloc[0] == 0) {
          zouHe->setZouHeVelBC2D_xn(envWidth, envWidth, ny-1-envWidth, 0.0, currentStep);
        }
        if (comm->myloc[0] == comm->procgrid[0]-1) {
          zouHe->setZouHeVelBC2D_xp(nx-1-envWidth, envWidth, ny-1-envWidth, 0.0, currentStep);
        }
        if (comm->myloc[1] == 0) {
          zouHe->setZouHeVelBC2D_yn(envWidth, envWidth, nx-1-envWidth, 0.0, currentStep);
        }
        if (comm->myloc[1] == comm->procgrid[1]-1) {
          zouHe->setZouHeVelBC2D_yp(ny-1-envWidth, envWidth, nx-1-envWidth, 0.0, currentStep);
        }
        break;
    }
  } else { // 3D
    switch (typeBC) {
      case 1: // Shear flow
        if (comm->myloc[1] == 0) {
          zouHe->setZouHeVelBC3D_yn(envWidth, 
                                     envWidth, nx-1-envWidth,
                                     envWidth, nz-1-envWidth,
                                     -u_infty, 0.0, 0.0, currentStep);
        }
        if (comm->myloc[1] == comm->procgrid[1]-1) {
          zouHe->setZouHeVelBC3D_yp(ny-1-envWidth, 
                                     envWidth, nx-1-envWidth,
                                     envWidth, nz-1-envWidth,
                                     u_infty, 0.0, 0.0, currentStep);
        }
        break;
        
      case 2: // X-direction flow
        if (comm->myloc[0] == 0) {
          zouHe->setZouHeVelBC3D_xn(envWidth, 
                                     envWidth, ny-1-envWidth,
                                     envWidth, nz-1-envWidth,
                                     u_infty, 0.0, 0.0, currentStep);
        }
        if (comm->myloc[0] == comm->procgrid[0]-1) {
          zouHe->setZouHeDensBC3D_xp(nx-1-envWidth, 
                                     envWidth, ny-1-envWidth,
                                     envWidth, nz-1-envWidth,
                                     rho_outlet, 0.0, 0.0, currentStep);
        }
        if (comm->myloc[1] == 0) {
          zouHe->setZouHeVelBC3D_yn(envWidth, 
                                     envWidth, nx-1-envWidth,
                                     envWidth, nz-1-envWidth,
                                     u_infty, 0.0, 0.0, currentStep);
        }
        if (comm->myloc[1] == comm->procgrid[1]-1) {
          zouHe->setZouHeVelBC3D_yp(ny-1-envWidth, 
                                     envWidth, nx-1-envWidth,
                                     envWidth, nz-1-envWidth,
                                     u_infty, 0.0, 0.0, currentStep);
        }
        if (comm->myloc[2] == 0) {
          zouHe->setZouHeVelBC3D_zn(envWidth, 
                                     envWidth, nx-1-envWidth,
                                     envWidth, ny-1-envWidth,
                                     u_infty, 0.0, 0.0, currentStep);
        }
        if (comm->myloc[2] == comm->procgrid[2]-1) {
          zouHe->setZouHeVelBC3D_zp(nz-1-envWidth, 
                                     envWidth, nx-1-envWidth,
                                     envWidth, ny-1-envWidth,
                                     u_infty, 0.0, 0.0, currentStep);
        }
        break;
        
      case 3: // Channel flow
        if (comm->myloc[1] == 0) {
          zouHe->setZouHeVelBC3D_yn(envWidth, 
                                     envWidth, nx-1-envWidth,
                                     envWidth, nz-1-envWidth,
                                     0.0, 0.0, 0.0, currentStep);
        }
        if (comm->myloc[1] == comm->procgrid[1]-1) {
          zouHe->setZouHeVelBC3D_yp(ny-1-envWidth, 
                                     envWidth, nx-1-envWidth,
                                     envWidth, nz-1-envWidth,
                                     0.0, 0.0, 0.0, currentStep);
        }
        break;
        
      case 4: // Channel with inlet/outlet
        if (comm->myloc[0] == 0) {
          zouHe->setZouHeVelBC3D_xn(envWidth, 
                                     envWidth, ny-1-envWidth,
                                     envWidth, nz-1-envWidth,
                                     u_infty, 0.0, 0.0, currentStep);
        }
        if (comm->myloc[0] == comm->procgrid[0]-1) {
          zouHe->setZouHeDensBC3D_xp(nx-1-envWidth, 
                                     envWidth, ny-1-envWidth,
                                     envWidth, nz-1-envWidth,
                                     rho_outlet, 0.0, 0.0, currentStep);
        }
        if (comm->myloc[1] == 0) {
          zouHe->setZouHeVelBC3D_yn(envWidth, 
                                     envWidth, nx-1-envWidth,
                                     envWidth, nz-1-envWidth,
                                     0.0, 0.0, 0.0, currentStep);
        }
        if (comm->myloc[1] == comm->procgrid[1]-1) {
          zouHe->setZouHeVelBC3D_yp(ny-1-envWidth, 
                                     envWidth, nx-1-envWidth,
                                     envWidth, nz-1-envWidth,
                                     0.0, 0.0, 0.0, currentStep);
        }
        if (comm->myloc[2] == 0) {
          zouHe->setZouHeVelBC3D_zn(envWidth, 
                                     envWidth, nx-1-envWidth,
                                     envWidth, ny-1-envWidth,
                                     0.0, 0.0, 0.0, currentStep);
        }
        if (comm->myloc[2] == comm->procgrid[2]-1) {
          zouHe->setZouHeVelBC3D_zp(nz-1-envWidth, 
                                     envWidth, nx-1-envWidth,
                                     envWidth, ny-1-envWidth,
                                     0.0, 0.0, 0.0, currentStep);
        }
        break;
        
      case 5: // Closed box
        if (comm->myloc[0] == 0) {
          zouHe->setZouHeVelBC3D_xn(envWidth, 
                                     envWidth, ny-1-envWidth,
                                     envWidth, nz-1-envWidth,
                                     0.0, 0.0, 0.0, currentStep);
        }
        if (comm->myloc[0] == comm->procgrid[0]-1) {
          zouHe->setZouHeVelBC3D_xp(nx-1-envWidth, 
                                     envWidth, ny-1-envWidth,
                                     envWidth, nz-1-envWidth,
                                     0.0, 0.0, 0.0, currentStep);
        }
        if (comm->myloc[1] == 0) {
          zouHe->setZouHeVelBC3D_yn(envWidth, 
                                     envWidth, nx-1-envWidth,
                                     envWidth, nz-1-envWidth,
                                     0.0, 0.0, 0.0, currentStep);
        }
        if (comm->myloc[1] == comm->procgrid[1]-1) {
          zouHe->setZouHeVelBC3D_yp(ny-1-envWidth, 
                                     envWidth, nx-1-envWidth,
                                     envWidth, nz-1-envWidth,
                                     0.0, 0.0, 0.0, currentStep);
        }
        if (comm->myloc[2] == 0) {
          zouHe->setZouHeVelBC3D_zn(envWidth, 
                                     envWidth, nx-1-envWidth,
                                     envWidth, ny-1-envWidth,
                                     0.0, 0.0, 0.0, currentStep);
        }
        if (comm->myloc[2] == comm->procgrid[2]-1) {
          zouHe->setZouHeVelBC3D_zp(nz-1-envWidth, 
                                     envWidth, nx-1-envWidth,
                                     envWidth, ny-1-envWidth,
                                     0.0, 0.0, 0.0, currentStep);
        }
        break;
    }
  }
}