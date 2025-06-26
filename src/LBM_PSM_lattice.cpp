/*------------------------------------------------------ 
This file is part of the LAMMPS-LBM-PSM project.

LAMMPS-LBM-PSM is an open-source project distributed
under the GNU General Public License.

See the README and License file in the top-level 
LAMMPS-LBM-PSM directory for more details.

Tim Najuch, 2022
------------------------------------------------------*/

#include "LBM_PSM_lattice.h"
#include <algorithm>
#include <cmath>
#include <mpi.h>

LBMPSMLattice::LBMPSMLattice(int nx_, int ny_, int nz_, int decomposition[3], 
                             int procCoordinates_[3], std::vector<double> origin_, 
                             std::vector<double> boxLength_, int dimension_)
  : envelopeWidth(1), dimension(dimension_), 
    origin_global(origin_), currentStep(0), nextStep(1)
{
  // Initialize processor coordinates
  for (int i = 0; i < 3; ++i) {
    procCoordinates[i] = procCoordinates_[i];
  }

  // Lambda for local dimension calculation
  auto calc_local_dim = [](int global, int decomp, int coord, int env) {
    const int base = global / decomp;
    const int remainder = global % decomp;
    if (remainder == 0) return base + 2 * env;
    return (coord == decomp - 1) ? (base + remainder + 2 * env) : (base + 2 * env);
  };

  // Calculate local grid dimensions
  nx = calc_local_dim(nx_, decomposition[0], procCoordinates[0], envelopeWidth);
  ny = calc_local_dim(ny_, decomposition[1], procCoordinates[1], envelopeWidth);
  nz = (dimension == 3) ? 
        calc_local_dim(nz_, decomposition[2], procCoordinates[2], envelopeWidth) : 
        1;
  
  q = (dimension == 3) ? 19 : 9;

  // Gather local dimensions across all processors
  const int num_procs = decomposition[0] * decomposition[1] * decomposition[2];
  nxLocal.resize(num_procs);
  nyLocal.resize(num_procs);
  nzLocal.resize(num_procs);
  
  MPI_Allgather(&nx, 1, MPI_INT, nxLocal.data(), 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(&ny, 1, MPI_INT, nyLocal.data(), 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(&nz, 1, MPI_INT, nzLocal.data(), 1, MPI_INT, MPI_COMM_WORLD);

  // Build grid views for each dimension
  nxLocalGrid.resize(decomposition[0]);
  for (int iproc = 0; iproc < decomposition[0]; ++iproc) {
    nxLocalGrid[iproc] = nxLocal[iproc * decomposition[1] * decomposition[2]];
  }

  nyLocalGrid.resize(decomposition[1]);
  for (int jproc = 0; jproc < decomposition[1]; ++jproc) {
    nyLocalGrid[jproc] = nyLocal[jproc * decomposition[2]];
  }

  nzLocalGrid.resize(decomposition[2]);
  for (int kproc = 0; kproc < decomposition[2]; ++kproc) {
    nzLocalGrid[kproc] = nzLocal[kproc];
  }

  // Precompute lattice constants
  cs = 1.0 / std::sqrt(3.0);
  csPow2 = cs * cs;
  csPow4 = csPow2 * csPow2;
  invCsPow2 = 1.0 / csPow2;
  invCsPow4 = 1.0 / csPow4;

  // Initialize lattice vectors
  if (q == 9) { // D2Q9
    e = {0,0,0,  1,0,0,  0,1,0,  -1,0,0,  0,-1,0,  
          1,1,0,  -1,1,0,  -1,-1,0,  1,-1,0};
    w = {4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 
         1.0/36, 1.0/36, 1.0/36, 1.0/36};
  } 
  else if (q == 19) { // D3Q19
    e = {0,0,0,  1,0,0, -1,0,0,  0,1,0, 0,-1,0, 0,0,1, 0,0,-1,
          1,1,0, -1,-1,0, 1,0,1, -1,0,-1, 0,1,1, 0,-1,-1,
          1,-1,0, -1,1,0, 1,0,-1, -1,0,1, 0,1,-1, 0,-1,1};
    w = {1.0/3, 1.0/18, 1.0/18, 1.0/18, 1.0/18, 1.0/18, 1.0/18,
         1.0/36, 1.0/36, 1.0/36, 1.0/36, 1.0/36, 1.0/36,
         1.0/36, 1.0/36, 1.0/36, 1.0/36, 1.0/36, 1.0/36};
  }

  // Allocate memory
  const size_t total_nodes = nx * ny * nz;
  f.resize(2 * total_nodes * q, 0.0);
  f0.resize(total_nodes * q, 0.0);
  
  rho.resize(total_nodes, 0.0);
  B.resize(total_nodes, 0.0);
  x.resize(total_nodes, 0.0);
  y.resize(total_nodes, 0.0);
  z.resize(total_nodes, 0.0);
  u.resize(3 * total_nodes, 0.0);
  us.resize(3 * total_nodes, 0.0);
  
  pData.resize(total_nodes);
}

LBMPSMLattice::~LBMPSMLattice() {}

void LBMPSMLattice::initialise_domain(double dx_, double dy_, double dz_) {
  dx = dx_;
  dy = dy_;
  dz = dz_;

  // Lambda to compute processor offset
  auto compute_offset = [](const std::vector<int>& localGrid, int coord, int env) {
    int offset = 0;
    for (int i = 0; i < coord; ++i) 
      offset += localGrid[i] - 2 * env;
    return offset;
  };

  // Compute processor-specific offsets
  const int nxOffset = compute_offset(nxLocalGrid, procCoordinates[0], envelopeWidth);
  const int nyOffset = compute_offset(nyLocalGrid, procCoordinates[1], envelopeWidth);
  const int nzOffset = (dimension == 3) ? 
        compute_offset(nzLocalGrid, procCoordinates[2], envelopeWidth) : 0;

  // Compute starting coordinates
  const double x0 = origin_global[0] + nxOffset * dx - envelopeWidth * dx;
  const double y0 = origin_global[1] + nyOffset * dy - envelopeWidth * dy;
  const double z0 = (dimension == 3) ? 
        (origin_global[2] + nzOffset * dz - envelopeWidth * dz) : 0.0;

  // Initialize node coordinates
  for (int i = 0; i < nx; ++i) {
    const double xpos = x0 + i * dx;
    for (int j = 0; j < ny; ++j) {
      const double ypos = y0 + j * dy;
      for (int k = 0; k < nz; ++k) {
        const int idx = i * ny * nz + j * nz + k;
        x[idx] = xpos;
        y[idx] = ypos;
        z[idx] = (dimension == 3) ? (z0 + k * dz) : 0.0;
        
        // Initialize particle data
        pData[idx].particleID = {0, 0};
        pData[idx].solidFraction = {0.0, 0.0};
        std::fill(pData[idx].particleVelocity.begin(), pData[idx].particleVelocity.end(), 0.0);
        std::fill(pData[idx].hydrodynamicForce.begin(), pData[idx].hydrodynamicForce.end(), 0.0);
      }
    }
  }
}

// Particle data management functions
void LBMPSMLattice::setParticleOnLattice(int index, LAMMPS_NS::tagint pID, 
                                         double uP[3], double eps) {
  auto& pd = pData[index];
  
  for (int slot = 0; slot < 2; ++slot) {
    if (pd.particleID[slot] == pID) {
      updateParticleSlot(pd, slot, pID, uP, eps);
      return;
    }
  }
  
  for (int slot = 0; slot < 2; ++slot) {
    if (pd.particleID[slot] == 0) {
      updateParticleSlot(pd, slot, pID, uP, eps);
      return;
    }
  }
}

void LBMPSMLattice::updateParticleSlot(ParticleDataOnLattice& pd, int slot,
                                      LAMMPS_NS::tagint pID, double uP[3], 
                                      double eps) {
  pd.particleID[slot] = pID;
  pd.solidFraction[slot] = eps;
  
  const int vel_offset = 3 * slot;
  pd.particleVelocity[vel_offset]     = uP[0];
  pd.particleVelocity[vel_offset + 1] = uP[1];
  pd.particleVelocity[vel_offset + 2] = uP[2];
  
  const int force_offset = 3 * slot;
  pd.hydrodynamicForce[force_offset]     = 0.0;
  pd.hydrodynamicForce[force_offset + 1] = 0.0;
  pd.hydrodynamicForce[force_offset + 2] = 0.0;
}

void LBMPSMLattice::setToZero(int index, LAMMPS_NS::tagint pID) {
  auto& pd = pData[index];
  for (int slot = 0; slot < 2; ++slot) {
    if (pd.particleID[slot] == pID) {
      resetParticleSlot(pd, slot);
      return;
    }
  }
}

void LBMPSMLattice::resetParticleSlot(ParticleDataOnLattice& pd, int slot) {
  pd.particleID[slot] = 0;
  pd.solidFraction[slot] = 0.0;
  
  const int vel_offset = 3 * slot;
  pd.particleVelocity[vel_offset]     = 0.0;
  pd.particleVelocity[vel_offset + 1] = 0.0;
  pd.particleVelocity[vel_offset + 2] = 0.0;
  
  const int force_offset = 3 * slot;
  pd.hydrodynamicForce[force_offset]     = 0.0;
  pd.hydrodynamicForce[force_offset + 1] = 0.0;
  pd.hydrodynamicForce[force_offset + 2] = 0.0;
}

// Accessors
int LBMPSMLattice::get_currentStep() { return currentStep; }
void LBMPSMLattice::set_currentStep(int currentStep_) { 
  currentStep = currentStep_; 
  nextStep = 1 - currentStep_; 
}

int LBMPSMLattice::get_nx() { return nx; }
int LBMPSMLattice::get_ny() { return ny; }
int LBMPSMLattice::get_nz() { return nz; }
int LBMPSMLattice::get_envelopeWidth() { return envelopeWidth; }
int LBMPSMLattice::get_q() { return q; }

int LBMPSMLattice::get_nxLocal(int iProcIndex) { 
  return nxLocalGrid[iProcIndex]; 
}

int LBMPSMLattice::get_nyLocal(int jProcIndex) { 
  return nyLocalGrid[jProcIndex]; 
}

int LBMPSMLattice::get_nzLocal(int kProcIndex) { 
  return nzLocalGrid[kProcIndex]; 
}

void LBMPSMLattice::set_B(int index, double B_) { B[index] = B_; }
double LBMPSMLattice::get_B(int index) { return B[index]; }
double LBMPSMLattice::get_rho(int index) { return rho[index]; }
double LBMPSMLattice::get_u(int index) { return u[index]; }

double LBMPSMLattice::get_u_at_node(int index_node_1D, int direction) { 
  return u[index_node_1D * 3 + direction]; 
}

std::vector<double>& LBMPSMLattice::getVector_f() { return f; }
void LBMPSMLattice::setVector_f(std::vector<double>& fcopy) { f = fcopy; }

ParticleDataOnLattice LBMPSMLattice::getParticleDataOnLatticeNode(int index) { 
  return pData[index]; 
}

double LBMPSMLattice::getSolidFractionOnLattice(int index, int slot) { 
  return pData[index].solidFraction[slot]; 
}

std::vector<double> LBMPSMLattice::getSolidVelocityOnLattice(int index) {
    return pData[index].particleVelocityAsVector();
}

std::vector<double> LBMPSMLattice::getSolidVelocityOnLattice(int index, int pID) {
    auto& pd = pData[index];
    for (int slot = 0; slot < 2; ++slot) {
        if (pd.particleID[slot] == pID) {
            const int offset = 3 * slot;
            return {
                pd.particleVelocity[offset], 
                pd.particleVelocity[offset+1], 
                pd.particleVelocity[offset+2]
            };
        }
    }
    return {0.0, 0.0, 0.0};
}

void LBMPSMLattice::add_Fhyd(int index, LAMMPS_NS::tagint pID, double Fhyd, int dir) {
  auto& pd = pData[index];
  for (int slot = 0; slot < 2; ++slot) {
    if (pd.particleID[slot] == pID) {
      pd.hydrodynamicForce[3 * slot + dir] += Fhyd;
      return;
    }
  }
}

std::vector<int> LBMPSMLattice::get_procCoordinates() {
  return {procCoordinates[0], procCoordinates[1], procCoordinates[2]};
}

// Vector accessors
std::vector<double> LBMPSMLattice::get_B() { return B; }
std::vector<double> LBMPSMLattice::get_rho() { return rho; }
std::vector<double> LBMPSMLattice::get_x() { return x; }
std::vector<double> LBMPSMLattice::get_y() { return y; }
std::vector<double> LBMPSMLattice::get_z() { return z; }
std::vector<double> LBMPSMLattice::get_u() { return u; }

std::vector<double>& LBMPSMLattice::get_B_reference() { return B; }
std::vector<double>& LBMPSMLattice::get_rho_reference() { return rho; }
std::vector<double>& LBMPSMLattice::get_x_reference() { return x; }
std::vector<double>& LBMPSMLattice::get_y_reference() { return y; }
std::vector<double>& LBMPSMLattice::get_z_reference() { return z; }
std::vector<double>& LBMPSMLattice::get_u_reference() { return u; }