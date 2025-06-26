/*------------------------------------------------------ 
This file is part of the LAMMPS-LBM-PSM project.

LAMMPS-LBM-PSM is an open-source project distributed
under the GNU General Public License.

See the README and License file in the top-level 
LAMMPS-LBM-PSM directory for more details.

Tim Najuch, 2022
------------------------------------------------------*/

#include "fix_LBM_PSM_writeVTK.h"
#include "fix_LBM_PSM.h"
#include "domain.h"
#include "comm.h"
#include "update.h"
#include "error.h"
#include <vector>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <cmath>

using namespace LAMMPS_NS;

WriteVTK::WriteVTK(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg) {
  if (narg < 5) error->all(FLERR,"Illegal fix lbm-psm-vtk command");

  for(int ifix=0; ifix<modify->nfix; ifix++)
    if(strcmp(modify->fix[ifix]->style,"lbm-psm")==0)
      fixLBMPSM = dynamic_cast<fix_LBM_PSM *>(modify->fix[ifix]);

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"every") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix lbm-psm-vtk command");
      nevery = atoi(arg[iarg+1]);
      if (nevery <= 0) error->all(FLERR,"Illegal fix lbm-psm-vtk command");
      iarg += 2;
    } else error->all(FLERR,"Illegal fix lbm-psm-vtk command");
  }
}

WriteVTK::~WriteVTK() {}

int WriteVTK::setmask() {
  int mask = 0;
  mask |= FixConst::PRE_FORCE;
  return mask;
}

void WriteVTK::init() {
  decomposition[0] = comm->procgrid[0];
  decomposition[1] = comm->procgrid[1];
  decomposition[2] = comm->procgrid[2];

  nx = fixLBMPSM->dynamics->get_nx();
  ny = fixLBMPSM->dynamics->get_ny();
  nz = fixLBMPSM->dynamics->get_nz();

  std::ostringstream timeStringTmp;  
  timeStringTmp << "flowField_" << std::setw(10) << std::setfill('0') << "0" << ".vtk";
  std::string timeString(timeStringTmp.str());

  write_vtk(timeString,
            fixLBMPSM->dynamics->get_x_reference(), 1.0,
            fixLBMPSM->dynamics->get_y_reference(), 1.0,
            fixLBMPSM->dynamics->get_z_reference(), 1.0,
            fixLBMPSM->dynamics->get_B_reference(), 1.0,
            fixLBMPSM->dynamics->get_rho_reference(), fixLBMPSM->get_rho(),
            fixLBMPSM->dynamics->get_u_reference(), 0.0);
}

void WriteVTK::pre_force(int) {
  if (update->ntimestep % nevery) return;
  const double u_infty = fixLBMPSM->unitConversion->get_u_lb();
  const double Uc = fixLBMPSM->unitConversion->get_Uc();

  std::ostringstream timeStringTmp;
  timeStringTmp << "flowField_" << std::setw(10) << std::setfill('0') 
               << update->ntimestep << ".vtk";
  std::string timeString(timeStringTmp.str());

  write_vtk(timeString,
            fixLBMPSM->dynamics->get_x_reference(), 1.0,
            fixLBMPSM->dynamics->get_y_reference(), 1.0,
            fixLBMPSM->dynamics->get_z_reference(), 1.0,
            fixLBMPSM->dynamics->get_B_reference(), 1.0,
            fixLBMPSM->dynamics->get_rho_reference(), fixLBMPSM->get_rho(),
            fixLBMPSM->dynamics->get_u_reference(), Uc/u_infty);
}

void WriteVTK::write_vtk(std::string name_, 
                         std::vector<double> &x_, double x0_,
                         std::vector<double> &y_, double y0_,
                         std::vector<double> &z_, double z0_,
                         std::vector<double> &B_, double B0_,
                         std::vector<double> &rho_, double rho0_,
                         std::vector<double> &u_, double u0_) {
  const int envWidth = 1;
  int nzLoopStart = 0;
  int nzLoopEnd = 1;
  if (domain->dimension == 3) {
    nzLoopStart = envWidth;
    nzLoopEnd = nz - envWidth;
  }

  std::vector<double> B_clip(nx * ny * nz, 0.0);
  for (int i = 0; i < nx*ny*nz; ++i) {
    const auto& pdata = fixLBMPSM->dynamics->getParticleDataOnLatticeNode(i);
    B_clip[i] = std::min(pdata.solidFraction[0] + pdata.solidFraction[1], 1.0);
  }

  const int nprocs = comm->nprocs;
  const int nelements = nx * ny * nz;
  const int nelements3D = nelements * 3;

  std::vector<int> MPIGathervCounts(nprocs, 0);
  std::vector<int> MPIGathervDispls(nprocs, 0);
  std::vector<int> MPIGathervCounts3D(nprocs, 0);
  std::vector<int> MPIGathervDispls3D(nprocs, 0);

  MPI_Gather(&nelements, 1, MPI_INT, MPIGathervCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&nelements3D, 1, MPI_INT, MPIGathervCounts3D.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (comm->me == 0) {
    for (int i = 1; i < nprocs; i++) {
      MPIGathervDispls[i] = MPIGathervDispls[i-1] + MPIGathervCounts[i-1];
      MPIGathervDispls3D[i] = MPIGathervDispls3D[i-1] + MPIGathervCounts3D[i-1];
    }
  }

  std::vector<double> x0, y0, z0, B0, rho0, u0;
  if (comm->me == 0) {
    const size_t totalSize = MPIGathervDispls.back() + MPIGathervCounts.back();
    x0.resize(totalSize);
    y0.resize(totalSize);
    z0.resize(totalSize);
    B0.resize(totalSize);
    rho0.resize(totalSize);
    u0.resize(MPIGathervDispls3D.back() + MPIGathervCounts3D.back());
  }

  MPI_Gatherv(x_.data(), nelements, MPI_DOUBLE, 
              x0.data(), MPIGathervCounts.data(), MPIGathervDispls.data(), 
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gatherv(y_.data(), nelements, MPI_DOUBLE, 
              y0.data(), MPIGathervCounts.data(), MPIGathervDispls.data(), 
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gatherv(z_.data(), nelements, MPI_DOUBLE, 
              z0.data(), MPIGathervCounts.data(), MPIGathervDispls.data(), 
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gatherv(B_clip.data(), nelements, MPI_DOUBLE, 
              B0.data(), MPIGathervCounts.data(), MPIGathervDispls.data(), 
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gatherv(rho_.data(), nelements, MPI_DOUBLE, 
              rho0.data(), MPIGathervCounts.data(), MPIGathervDispls.data(), 
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gatherv(u_.data(), nelements3D, MPI_DOUBLE, 
              u0.data(), MPIGathervCounts3D.data(), MPIGathervDispls3D.data(), 
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (comm->me == 0) {
    auto scale_vector = [](std::vector<double> &vec, double scaling) {
      std::transform(vec.begin(), vec.end(), vec.begin(),
                     [scaling](double val) { return val * scaling; });
    };

    scale_vector(x0, x0_);
    scale_vector(y0, y0_);
    scale_vector(z0, z0_);
    scale_vector(rho0, rho0_);
    scale_vector(u0, u0_);

    int nxTotal = 0, nyTotal = 0, nzTotal = 0;
    for (int i = 0; i < decomposition[0]; i++) 
      nxTotal += fixLBMPSM->dynamics->get_nxLocal(i);
    for (int j = 0; j < decomposition[1]; j++) 
      nyTotal += fixLBMPSM->dynamics->get_nyLocal(j);
    for (int k = 0; k < decomposition[2]; k++) 
      nzTotal += fixLBMPSM->dynamics->get_nzLocal(k);

    std::ofstream ovel(name_);
    ovel << "# vtk DataFile Version 3.1\n";
    ovel << "Fluid flow field from LBM simulation\n";
    ovel << "ASCII\n";
    ovel << "DATASET RECTILINEAR_GRID\n";
    
    if (domain->dimension == 2) {
      ovel << "DIMENSIONS " << nxTotal - 2*envWidth*decomposition[0] << " " 
           << nyTotal - 2*envWidth*decomposition[1] << " 1\n";
    } else {
      ovel << "DIMENSIONS " << nxTotal - 2*envWidth*decomposition[0] << " " 
           << nyTotal - 2*envWidth*decomposition[1] << " " 
           << nzTotal - 2*envWidth*decomposition[2] << "\n";
    }

    // Write coordinates using correct indexing
    ovel << "X_COORDINATES " << nxTotal - 2*envWidth*decomposition[0] << " float\n";
    for (int iproc = 0; iproc < decomposition[0]; iproc++) {
      const int nxLocal = fixLBMPSM->dynamics->get_nxLocal(iproc);
      const int nyLocal = fixLBMPSM->dynamics->get_nyLocal(0);
      const int nzLocal = (domain->dimension == 3) ? fixLBMPSM->dynamics->get_nzLocal(0) : 1;
      for (int i = envWidth; i < nxLocal - envWidth; i++) {
        const int j = envWidth;
        const int k = (domain->dimension == 3) ? envWidth : 0;
        const int procIndex = iproc * decomposition[1] * decomposition[2];
        const int index = i*nyLocal*nzLocal + j*nzLocal + k + MPIGathervDispls[procIndex];
        ovel << x0[index] << "\n";
      }
    }

    ovel << "Y_COORDINATES " << nyTotal - 2*envWidth*decomposition[1] << " float\n";
    for (int jproc = 0; jproc < decomposition[1]; jproc++) {
      const int nyLocal = fixLBMPSM->dynamics->get_nyLocal(jproc);
      const int nxLocal = fixLBMPSM->dynamics->get_nxLocal(0);
      const int nzLocal = (domain->dimension == 3) ? fixLBMPSM->dynamics->get_nzLocal(0) : 1;
      for (int j = envWidth; j < nyLocal - envWidth; j++) {
        const int i = envWidth;
        const int k = (domain->dimension == 3) ? envWidth : 0;
        const int procIndex = jproc * decomposition[2];
        const int index = i*nyLocal*nzLocal + j*nzLocal + k + MPIGathervDispls[procIndex];
        ovel << y0[index] << "\n";
      }
    }

    ovel << "Z_COORDINATES " << ((domain->dimension == 2) ? 1 : nzTotal - 2*envWidth*decomposition[2]) 
         << " float\n";
    if (domain->dimension == 2) {
      ovel << "0.0\n";
    } else {
      for (int kproc = 0; kproc < decomposition[2]; kproc++) {
        const int nzLocal = fixLBMPSM->dynamics->get_nzLocal(kproc);
        const int nxLocal = fixLBMPSM->dynamics->get_nxLocal(0);
        const int nyLocal = fixLBMPSM->dynamics->get_nyLocal(0);
        for (int k = envWidth; k < nzLocal - envWidth; k++) {
          const int i = envWidth;
          const int j = envWidth;
          const int procIndex = kproc;
          const int index = i*nyLocal*nzLocal + j*nzLocal + k + MPIGathervDispls[procIndex];
          ovel << z0[index] << "\n";
        }
      }
    }

    // Point data
    const int nPoints = (domain->dimension == 2) ? 
        (nxTotal - 2*envWidth*decomposition[0]) * (nyTotal - 2*envWidth*decomposition[1]) :
        (nxTotal - 2*envWidth*decomposition[0]) * (nyTotal - 2*envWidth*decomposition[1]) * 
        (nzTotal - 2*envWidth*decomposition[2]);
    ovel << "\nPOINT_DATA " << nPoints << "\n";

    // Solid fraction
    ovel << "SCALARS SolidFraction FLOAT\nLOOKUP_TABLE default\n";
    for (int kproc = 0; kproc < decomposition[2]; kproc++) {
      const int nzLocal = fixLBMPSM->dynamics->get_nzLocal(kproc);
      const int kStart = (domain->dimension == 3) ? envWidth : 0;
      const int kEnd = (domain->dimension == 3) ? nzLocal - envWidth : 1;
      for (int k = kStart; k < kEnd; k++) {
        for (int jproc = 0; jproc < decomposition[1]; jproc++) {
          const int nyLocal = fixLBMPSM->dynamics->get_nyLocal(jproc);
          for (int j = envWidth; j < nyLocal - envWidth; j++) {
            for (int iproc = 0; iproc < decomposition[0]; iproc++) {
              const int nxLocal = fixLBMPSM->dynamics->get_nxLocal(iproc);
              for (int i = envWidth; i < nxLocal - envWidth; i++) {
                const int procIndex = iproc*decomposition[1]*decomposition[2] 
                                    + jproc*decomposition[2] 
                                    + kproc;
                const int index = i*nyLocal*nzLocal + j*nzLocal + k + MPIGathervDispls[procIndex];
                ovel << B0[index] << "\n";
              }
            }
          }
        }
      }
    }

    // Density
    ovel << "\nSCALARS Density FLOAT\nLOOKUP_TABLE default\n";
    for (int kproc = 0; kproc < decomposition[2]; kproc++) {
      const int nzLocal = fixLBMPSM->dynamics->get_nzLocal(kproc);
      const int kStart = (domain->dimension == 3) ? envWidth : 0;
      const int kEnd = (domain->dimension == 3) ? nzLocal - envWidth : 1;
      for (int k = kStart; k < kEnd; k++) {
        for (int jproc = 0; jproc < decomposition[1]; jproc++) {
          const int nyLocal = fixLBMPSM->dynamics->get_nyLocal(jproc);
          for (int j = envWidth; j < nyLocal - envWidth; j++) {
            for (int iproc = 0; iproc < decomposition[0]; iproc++) {
              const int nxLocal = fixLBMPSM->dynamics->get_nxLocal(iproc);
              for (int i = envWidth; i < nxLocal - envWidth; i++) {
                const int procIndex = iproc*decomposition[1]*decomposition[2] 
                                    + jproc*decomposition[2] 
                                    + kproc;
                const int index = i*nyLocal*nzLocal + j*nzLocal + k + MPIGathervDispls[procIndex];
                ovel << rho0[index] << "\n";
              }
            }
          }
        }
      }
    }

    // Velocity
    ovel << "\nVECTORS Velocity FLOAT\n";
    for (int kproc = 0; kproc < decomposition[2]; kproc++) {
      const int nzLocal = fixLBMPSM->dynamics->get_nzLocal(kproc);
      const int kStart = (domain->dimension == 3) ? envWidth : 0;
      const int kEnd = (domain->dimension == 3) ? nzLocal - envWidth : 1;
      for (int k = kStart; k < kEnd; k++) {
        for (int jproc = 0; jproc < decomposition[1]; jproc++) {
          const int nyLocal = fixLBMPSM->dynamics->get_nyLocal(jproc);
          for (int j = envWidth; j < nyLocal - envWidth; j++) {
            for (int iproc = 0; iproc < decomposition[0]; iproc++) {
              const int nxLocal = fixLBMPSM->dynamics->get_nxLocal(iproc);
              for (int i = envWidth; i < nxLocal - envWidth; i++) {
                const int procIndex = iproc*decomposition[1]*decomposition[2] 
                                    + jproc*decomposition[2] 
                                    + kproc;
                const int index3D = (i*nyLocal*nzLocal + j*nzLocal + k) * 3 + MPIGathervDispls3D[procIndex];
                if (domain->dimension == 2) {
                  ovel << u0[index3D] << " " << u0[index3D+1] << " 0\n";
                } else {
                  ovel << u0[index3D] << " " << u0[index3D+1] << " " << u0[index3D+2] << "\n";
                }
              }
            }
          }
        }
      }
    }
    ovel.close();
  }
}

void WriteVTK::scale_vector(std::vector<double> &vec_, double scaling_) {
  std::transform(vec_.begin(), vec_.end(), vec_.begin(),
                 [scaling_](double val) { return val * scaling_; });
}