/*------------------------------------------------------ 
This file is part of the LAMMPS-LBM-PSM project.

LAMMPS-LBM-PSM is an open-source project distributed
under the GNU General Public License.

See the README and License file in the top-level 
LAMMPS-LBM-PSM directory for more details.

Tim Najuch, 2022
------------------------------------------------------*/

#ifndef LBM_PSM_LATTICE_2D_H
#define LBM_PSM_LATTICE_2D_H

#include <vector>
#include <array>
#include "lmptype.h"
#include "LBM_PSM_particleDataOnLattice.h"

class ZouHeBC;  // Forward declaration

class LBMPSMLattice {
protected:
    int nx, ny, nz;
    std::vector<int> nxLocal;
    std::vector<int> nyLocal;
    std::vector<int> nzLocal;
    std::vector<int> nxLocalGrid;
    std::vector<int> nyLocalGrid;
    std::vector<int> nzLocalGrid;
    int envelopeWidth;
    int dimension;
    int q;
    std::vector<double> w;
    std::vector<double> e;
    double cs, csPow2, csPow4;
    double invCsPow2, invCsPow4;
    double dx, dy, dz;
    int currentStep, nextStep;
    std::vector<double> f;
    std::vector<double> f0;
    std::vector<double> origin_global;
    std::vector<double> x, y, z;
    std::vector<double> rho;
    std::vector<double> B;
    std::vector<double> u;
    std::vector<double> us;
    std::vector<ParticleDataOnLattice> pData;
    
    // Processor coordinates storage
    std::array<int, 3> procCoordinates;

    // Index calculations
    inline int index_1D(int i, int j, int k) { 
        return i * ny * nz + j * nz + k; 
    }
    
    inline int index_2D(int i, int j, int k, int direction) { 
        return (i * ny * nz + j * nz + k) * 3 + direction; 
    }
    
    inline int index_fi(int i, int j, int k, int iq, int step) { 
        return (nx * ny * nz * q * step) + (i * ny * nz + j * nz + k) * q + iq; 
    }

    // Helper functions
    void updateParticleSlot(ParticleDataOnLattice& pd, int slot, LAMMPS_NS::tagint pID, double uP[3], double eps);
    void resetParticleSlot(ParticleDataOnLattice& pd, int slot);

public:
    LBMPSMLattice(int nx_, int ny_, int nz_, int decomposition[3], int procCoordinates_[3], 
                 std::vector<double> origin_, std::vector<double> boxLength_, int dimension_);
    ~LBMPSMLattice();

    void initialise_domain(double dx_, double dy_, double dz_);

    int get_currentStep();
    void set_currentStep(int currentStep);

    // Population accessors
    void set_f(int i_, int j_, int k_, int iq_, int step_, double value_) {
        f[index_fi(i_, j_, k_, iq_, step_)] = value_;
    }
    
    void set_f(int ind_iq_, double value_) {
        f[ind_iq_] = value_;
    }
    
    double get_f(int i_, int j_, int k_, int iq_, int step_) {
        return f[index_fi(i_, j_, k_, iq_, step_)];
    }
    
    double get_f(int ind_iq_) {
        return f[ind_iq_];
    }
    
    void set_f0(int i_, int j_, int k_, int iq_, double value_) {
        f0[index_fi(i_, j_, k_, iq_, 0)] = value_;
    }
    
    void set_f0(int ind_iq_, double value_) {
        f0[ind_iq_] = value_;
    }
    
    double get_f0(int i_, int j_, int k_, int iq_) {
        return f0[index_fi(i_, j_, k_, iq_, 0)];
    }
    
    double get_f0(int ind_iq_) {
        return f0[ind_iq_];
    }

    // Vector accessors
    std::vector<double> get_B();
    std::vector<double> get_rho();
    std::vector<double> get_x();
    std::vector<double> get_y();
    std::vector<double> get_z();
    std::vector<double> get_u();

    std::vector<double>& get_B_reference();
    std::vector<double>& get_rho_reference();
    std::vector<double>& get_x_reference();
    std::vector<double>& get_y_reference();
    std::vector<double>& get_z_reference();
    std::vector<double>& get_u_reference();

    int get_nx();
    int get_ny();
    int get_nz();
    int get_envelopeWidth();
    int get_q();

    int get_nxLocal(int iProcIndex);
    int get_nyLocal(int jProcIndex);
    int get_nzLocal(int kProcIndex);

    void set_B(int index, double B_);
    double get_B(int index);
    double get_rho(int index);
    double get_u(int index);
    double get_u_at_node(int index_node_1D, int direction);

    std::vector<double>& getVector_f();
    void setVector_f(std::vector<double>& fcopy);

    ParticleDataOnLattice getParticleDataOnLatticeNode(int index);
    void setParticleOnLattice(int index, LAMMPS_NS::tagint pID, double uP[3], double eps);
    void setToZero(int index, LAMMPS_NS::tagint pID);
    double getSolidFractionOnLattice(int index, int slot);
    std::vector<double> getSolidVelocityOnLattice(int index);
    std::vector<double> getSolidVelocityOnLattice(int index, int pID);
    void add_Fhyd(int index, LAMMPS_NS::tagint pID, double Fhyd, int dir);

    std::vector<int> get_procCoordinates();

    friend class ZouHeBC;
};

#endif