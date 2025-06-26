/*------------------------------------------------------ 
This file is part of the LAMMPS-LBM-PSM project.

LAMMPS-LBM-PSM is an open-source project distributed
under the GNU General Public License.

See the README and License file in the top-level 
LAMMPS-LBM-PSM directory for more details.

Tim Najuch, 2022
------------------------------------------------------*/

#ifndef LBM_PSM_EXC_PART_DATA_H
#define LBM_PSM_EXC_PART_DATA_H

#include <vector>
#include "LBM_PSM_lattice.h"
#include "LBM_PSM_unitConversion.h"

class ExchangeParticleData {
private: 
    const int dimension;
    const std::vector<double> origin;
    const double sqrt2half_2D;
    const double sqrt2half_3D;

public:
    ExchangeParticleData(int dimension_, const std::vector<double>& origin_);
    ~ExchangeParticleData() = default;

    void setParticlesOnLattice(LBMPSMLattice *lattice_, UnitConversion *unitConversion, 
                               int numberParticles, LAMMPS_NS::tagint *tag, 
                               double **xPart, double **uPart, 
                               double **omega, double *rp);
    
    double calcSolidFraction(int i, int j, int k, double xP_LB, double yP_LB, 
                             double zP_LB, double rP_LB) const;
    
    void calculateHydrodynamicInteractions(LBMPSMLattice *lattice_, 
                                          UnitConversion *unitConversion, 
                                          LAMMPS_NS::tagint tag, double *xPart, 
                                          double rp, std::vector<double> &fHydro, 
                                          std::vector<double> &tHydro, 
                                          std::vector<double> &stresslet);
};

#endif