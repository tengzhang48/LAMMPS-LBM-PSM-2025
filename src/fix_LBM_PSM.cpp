/*------------------------------------------------------ 
This file is part of the LAMMPS-LBM-PSM project.

LAMMPS-LBM-PSM is an open-source project distributed
under the GNU General Public License.

See the README and License file in the top-level 
LAMMPS-LBM-PSM directory for more details.

Tim Najuch, 2022
------------------------------------------------------*/

#include "fix_LBM_PSM.h"
#include "atom.h"
#include "domain.h"
#include "update.h"
#include "error.h"
#include "comm.h"
#include "modify.h"
#include "utils.h"
#include "memory.h"
#include "fmt/core.h"
#include <vector>
#include <cmath>
#include <algorithm>

using namespace LAMMPS_NS;

fix_LBM_PSM::fix_LBM_PSM(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg) {
  if (narg < 15) error->all(FLERR,"Illegal fix lbm-psm command");

  virial_global_flag = virial_peratom_flag = 1;
  thermo_virial = 1;
  comm_reverse = 6;

  tau = 0.7; // default value for BGK relaxation parameter
  F_ext = std::vector<double>(3, 0.0);

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"every") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix lbm-psm command");
      nevery = atoi(arg[iarg+1]);
      if (nevery <= 0) error->all(FLERR,"Illegal fix lbm-psm command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"Nlc") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix lbm-psm command");
      Nlc = atoi(arg[iarg+1]);
      if (Nlc <= 0) error->all(FLERR,"Illegal fix lbm-psm command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"lc") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix lbm-psm command");
      lc = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (lc <= 0.0) error->all(FLERR,"Illegal fix lbm-psm command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"rho") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix lbm-psm command");
      rho = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (rho <= 0.0) error->all(FLERR,"Illegal fix lbm-psm command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"nu") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix lbm-psm command");
      nu = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (nu <= 0.0) error->all(FLERR,"Illegal fix lbm-psm command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"Re") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix lbm-psm command");
      Re = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (Re <= 0.0) error->all(FLERR,"Illegal fix lbm-psm command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"tau") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix lbm-psm command");
      tau = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (tau <= 0.5) error->all(FLERR,"Illegal fix lbm-psm command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"Fext") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix lbm-psm command");
      F_ext[0] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      F_ext[1] = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      F_ext[2] = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      iarg += 4;
    } else error->all(FLERR,"Illegal fix lbm-psm command");
  }

  // Allocate memory for storage of forces
  hydrodynamicInteractions = nullptr;
  grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);

  for (int i=0; i<atom->nmax; i++){
    for (int j=0; j<6; j++){
      hydrodynamicInteractions[i][j] = 0.0;
    }
  }
}

fix_LBM_PSM::~fix_LBM_PSM() {
  delete dynamics;
  delete unitConversion;
  delete exchangeParticleData;
  delete lbmmpicomm;
  memory->destroy(hydrodynamicInteractions);
}

int fix_LBM_PSM::setmask() {
  int mask = 0;
  mask |= FixConst::POST_FORCE;  // CORRECTED: Use scoped constant
  return mask;
}

void fix_LBM_PSM::init() {
  // Get processor grid information
  int decomposition[3] = {comm->procgrid[0], comm->procgrid[1], comm->procgrid[2]};
  int procCoordinates[3] = {comm->myloc[0], comm->myloc[1], comm->myloc[2]};
  int procNeigh[6] = {
    comm->procneigh[0][0], comm->procneigh[0][1],
    comm->procneigh[1][0], comm->procneigh[1][1],
    comm->procneigh[2][0], comm->procneigh[2][1]
  };

  // Create MPI communicator for LBM
  lbmmpicomm = new LBMPSMMPI(world, decomposition, procNeigh, procCoordinates, domain->dimension);
  
  // Set up unit conversion
  unitConversion = new UnitConversion(rho, nu, lc, Re, Nlc, tau, domain->dimension);
  
  // Calculate and set DEM timestep
  update->dt = unitConversion->get_phys_time(1.0)/(static_cast<double>(nevery));

  // Check Mach number stability
  const double Ma = unitConversion->get_u_lb()/(1.0/sqrt(3.0));
  if (Ma > 0.3) {
    error->all(FLERR,"Mach number > 0.3. Choose parameters for Ma < 0.3 stability");
  } else if (comm->me == 0) {
    std::string mesg = fmt::format("Mach number is Ma = {:.2f}\n", Ma);
    utils::logmesg(lmp, mesg);
    mesg = fmt::format("DEM timestep is dt = {:.6f}\n", update->dt);
    utils::logmesg(lmp, mesg);
  }

  // Domain size validation - FIXED: use original calculation method
  double SMALL = 1e-15;
  double dx = unitConversion->get_dx();
  
  // Verify domain lengths are multiples of cell width (original method)
  double xdiff = domain->xprd/dx - static_cast<int>(domain->xprd/dx + 0.5);
  double ydiff = domain->yprd/dx - static_cast<int>(domain->yprd/dx + 0.5);
  double zdiff = 0.0;
  if (domain->dimension == 3) {
    zdiff = domain->zprd/dx - static_cast<int>(domain->zprd/dx + 0.5);
  }

  if (xdiff*xdiff > SMALL || ydiff*ydiff > SMALL || 
      (domain->dimension == 3 && zdiff*zdiff > SMALL)) {
    error->all(FLERR, "Domain length must be integer multiple of cell width");
  }

  // Calculate lattice dimensions - FIXED: revert to original calculation
  int nx = static_cast<int>(domain->xprd/dx + 1.5);
  int ny = static_cast<int>(domain->yprd/dx + 1.5);
  int nz = 0;
  if (domain->dimension == 3) {
    nz = static_cast<int>(domain->zprd/dx + 1.5);
  }
  
  // Adjust for periodic boundaries (original method)
  if (domain->xperiodic) nx--;
  if (domain->yperiodic) ny--;
  if (domain->dimension == 3 && domain->zperiodic) nz--;

  // Set up domain parameters
  std::vector<double> boxLength = {domain->xprd, domain->yprd, domain->zprd};
  std::vector<double> origin = {domain->boxlo[0], domain->boxlo[1], domain->boxlo[2]};
  if (domain->dimension == 2) origin[2] = 0.0;

  // Convert external force to lattice units
  std::vector<double> F_lbm = unitConversion->get_volume_force_lb(F_ext);
  
  // Initialize LBM dynamics
  dynamics = new LBMPSMBGKDynamics(tau, nx, ny, nz, F_lbm, decomposition, 
                                  procCoordinates, origin, boxLength, domain->dimension);

  // Set up lattice and dynamics - FIXED: use actual dx values
  dynamics->initialise_domain(dx, dx, dx);
  dynamics->initialise_dynamics(1.0, 0.0, 0.0, 0.0);
  
  // Initialize particle exchange
  exchangeParticleData = new ExchangeParticleData(domain->dimension, origin);

  // Reset hydrodynamic interactions
  for (int i=0; i<atom->nmax; i++) {
    for (int j=0; j<6; j++) {
      hydrodynamicInteractions[i][j] = 0.0;
    }
  }
}

void fix_LBM_PSM::post_force(int vflag) {
  double **x = atom->x;
  double **v = atom->v;
  double **omega = atom->omega;
  double **f = atom->f;
  double **t = atom->torque;
  double *radius = atom->radius;
  tagint *tag = atom->tag;

  if (update->ntimestep % nevery) {
    // Apply stored forces if not a coupling step
    const int nlocal = atom->nlocal;
    for (int i = 0; i < nlocal; i++) {
      f[i][0] += hydrodynamicInteractions[i][0];
      f[i][1] += hydrodynamicInteractions[i][1];
      f[i][2] += hydrodynamicInteractions[i][2];
      t[i][0] += hydrodynamicInteractions[i][3];
      t[i][1] += hydrodynamicInteractions[i][4];
      t[i][2] += hydrodynamicInteractions[i][5];
    }
  } else {
    // Coupling step: perform LBM-DEM interaction
    comm->forward_comm();

    const int nPart = atom->nlocal + atom->nghost;
    std::vector<double> boxLength{domain->xprd, domain->yprd, domain->zprd};
    std::vector<double> origin{domain->boxlo[0], domain->boxlo[1], domain->boxlo[2]};

    // Map particles to lattice
    exchangeParticleData->setParticlesOnLattice(dynamics, unitConversion, nPart, 
                                               tag, x, v, omega, radius);

    const int envWidth = dynamics->get_envelopeWidth();
    // Communicate boundary data
    if (domain->dimension == 2) {
      lbmmpicomm->sendRecvData<double>(dynamics->getVector_f(), 0, dynamics->get_nx(), 
                                      dynamics->get_ny(), 1, envWidth, domain->xperiodic, 
                                      dynamics->get_currentStep());
      lbmmpicomm->sendRecvData<double>(dynamics->getVector_f(), 1, dynamics->get_nx(), 
                                      dynamics->get_ny(), 1, envWidth, domain->yperiodic, 
                                      dynamics->get_currentStep());
    } else {
      lbmmpicomm->sendRecvData<double>(dynamics->getVector_f(), 0, dynamics->get_nx(), 
                                      dynamics->get_ny(), dynamics->get_nz(), envWidth, 
                                      domain->xperiodic, dynamics->get_currentStep());
      lbmmpicomm->sendRecvData<double>(dynamics->getVector_f(), 1, dynamics->get_nx(), 
                                      dynamics->get_ny(), dynamics->get_nz(), envWidth, 
                                      domain->yperiodic, dynamics->get_currentStep());
      lbmmpicomm->sendRecvData<double>(dynamics->getVector_f(), 2, dynamics->get_nx(), 
                                      dynamics->get_ny(), dynamics->get_nz(), envWidth, 
                                      domain->zperiodic, dynamics->get_currentStep());
    }

    // Perform LBM collision and streaming
    dynamics->macroCollideStream();
    v_init(vflag);

    const int nlocal = atom->nlocal;
    const int nPartTotal = atom->nlocal + atom->nghost;
    std::vector<double> fh(3, 0.0);
    std::vector<double> th(3, 0.0);
    std::vector<double> stresslet(6, 0.0);

    // Calculate hydrodynamic forces
    for (int i = 0; i < nPartTotal; i++) {
      std::fill(fh.begin(), fh.end(), 0.0);
      std::fill(th.begin(), th.end(), 0.0);
      std::fill(stresslet.begin(), stresslet.end(), 0.0);

      exchangeParticleData->calculateHydrodynamicInteractions(dynamics, unitConversion, 
                                                            tag[i], x[i], radius[i], 
                                                            fh, th, stresslet);

      if (i < nlocal) {
        // Apply forces to local particles
        f[i][0] += fh[0];
        f[i][1] += fh[1];
        f[i][2] += fh[2];
        t[i][0] += th[0];
        t[i][1] += th[1];
        t[i][2] += th[2];
      } else {
        // Ghost particles only get forces for communication
        f[i][0] = fh[0];
        f[i][1] = fh[1];
        f[i][2] = fh[2];
        t[i][0] = th[0];
        t[i][1] = th[1];
        t[i][2] = th[2];
      }

      // Store hydrodynamic interactions
      hydrodynamicInteractions[i][0] = fh[0];
      hydrodynamicInteractions[i][1] = fh[1];
      hydrodynamicInteractions[i][2] = fh[2];
      hydrodynamicInteractions[i][3] = th[0];
      hydrodynamicInteractions[i][4] = th[1];
      hydrodynamicInteractions[i][5] = th[2];

      // Add stresslet contribution to virial (for all particles)
      double stresslet_arr[6] = {stresslet[0], stresslet[1], stresslet[2],
                                stresslet[3], stresslet[4], stresslet[5]};
      v_tally(i, stresslet_arr);
    }

    // Communicate forces - FIXED: use modern LAMMPS 2025 communication
    comm->reverse_comm();
    comm->reverse_comm(this); 
  }
}

double fix_LBM_PSM::get_rho() {
  return rho;
}

void fix_LBM_PSM::grow_arrays(int nmax) {
  memory->grow(hydrodynamicInteractions, nmax, 6, "fix_LBM_PSM:hydrodynamicInteractions");
}

void fix_LBM_PSM::copy_arrays(int i, int j, int /*delflag*/) {
  for (int k = 0; k < 6; k++) {
    hydrodynamicInteractions[j][k] = hydrodynamicInteractions[i][k];
  }
}

int fix_LBM_PSM::pack_exchange(int i, double *buf) {
  for (int k = 0; k < 6; k++) {
    buf[k] = hydrodynamicInteractions[i][k];
  }
  return 6;
}

int fix_LBM_PSM::unpack_exchange(int nlocal, double *buf) {
  for (int k = 0; k < 6; k++) {
    hydrodynamicInteractions[nlocal][k] = buf[k];
  }
  return 6;
}

int fix_LBM_PSM::pack_reverse_comm_size(int n, int first) {
  return n * 6;
}

int fix_LBM_PSM::pack_reverse_comm(int n, int first, double *buf) {
  int m = 0;
  for (int i = first; i < first + n; i++) {
    for (int k = 0; k < 6; k++) {
      buf[m++] = hydrodynamicInteractions[i][k];
    }
  }
  return m;
}

void fix_LBM_PSM::unpack_reverse_comm(int n, int *list, double *buf) {
  int m = 0;
  for (int i = 0; i < n; i++) {
    const int j = list[i];
    for (int k = 0; k < 6; k++) {
      hydrodynamicInteractions[j][k] += buf[m++];
    }
  }
}