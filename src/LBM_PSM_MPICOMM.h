/*------------------------------------------------------ 
This file is part of the LAMMPS-LBM-PSM project.

LAMMPS-LBM-PSM is an open-source project distributed
under the GNU General Public License.

See the README and License file in the top-level 
LAMMPS-LBM-PSM directory for more details.

Tim Najuch, 2022
------------------------------------------------------*/

#ifndef LBM_PSM_MPICOMM_H
#define LBM_PSM_MPICOMM_H

#include <vector>
#include <iostream>    // Added for std::cerr
#include <type_traits> // Added for std::is_same
#include "mpi.h"

class LBMPSMMPI {
public:
    LBMPSMMPI(MPI_Comm world_, int decomposition[3], int procNeigh[6], int procCoordinates_[3], int dimension_);
    ~LBMPSMMPI();
    
    int size;
    int rank;

    int dimensions[3];
    int procCoordinates[3];
    int northRank, eastRank, southRank, westRank, upRank, downRank;

    MPI_Comm world;
    MPI_Status status;

    template<typename T> 
    void sendRecvData(std::vector<T> &data_, int commDirection, int nx, int ny, int nz, int envelopeWidth, bool isPeriodic, int currentStep);

private:
    int dimension;
    int q;

    template<typename T> MPI_Datatype get_type();
    template<typename T> 
    void packData(std::vector<T> &sendBuf, const std::vector<T> &data, const int direction[3],
                 int envelopeStart, int dataSize, int nx, int ny, int nz, int envelopeWidth, int currentStep);
    
    template<typename T> 
    void unpackData(const std::vector<T> &recvBuf, std::vector<T> &data, const int direction[3],
                   int envelopeStart, int dataSize, int nx, int ny, int nz, int envelopeWidth, int currentStep);
};

template<typename T> 
MPI_Datatype LBMPSMMPI::get_type() {
    if (std::is_same<T, int>::value) return MPI_INT;
    if (std::is_same<T, float>::value) return MPI_FLOAT;
    if (std::is_same<T, double>::value) return MPI_DOUBLE;
    if (std::is_same<T, char>::value) return MPI_CHAR;
    if (std::is_same<T, short>::value) return MPI_SHORT;
    if (std::is_same<T, long>::value) return MPI_LONG;
    if (std::is_same<T, unsigned>::value) return MPI_UNSIGNED;
    if (std::is_same<T, unsigned long>::value) return MPI_UNSIGNED_LONG;
    
    std::cerr << "Error: Unsupported data type for MPI communication\n";
    MPI_Abort(world, 1);
    return MPI_DATATYPE_NULL;
}

template<typename T> 
void LBMPSMMPI::sendRecvData(std::vector<T> &data_, int commDirection, int nx, int ny, int nz, int envelopeWidth, bool isPeriodic, int currentStep) {
    const int dataSize = q;
    int commDataSize = 0;
    int direction[3] = {0, 0, 0};
    const int sendRank[2] = {commDirection == 0 ? westRank : 
                            (commDirection == 1 ? southRank : downRank),
                          commDirection == 0 ? eastRank : 
                            (commDirection == 1 ? northRank : upRank)};
    const int recvRank[2] = {commDirection == 0 ? eastRank : 
                            (commDirection == 1 ? northRank : upRank),
                          commDirection == 0 ? westRank : 
                            (commDirection == 1 ? southRank : downRank)};

    // Precompute communication data size
    switch(commDirection) {
        case 0:  // x-direction
            commDataSize = dataSize * envelopeWidth * (dimension == 2 ? ny : ny*nz);
            direction[0] = 1;
            break;
        case 1:  // y-direction
            commDataSize = dataSize * envelopeWidth * (dimension == 2 ? nx : nx*nz);
            direction[1] = 1;
            break;
        case 2:  // z-direction
            commDataSize = dataSize * envelopeWidth * nx*ny;
            direction[2] = 1;
            break;
    }

    MPI_Datatype commDataType = get_type<T>();
    std::vector<T> sendBuf(commDataSize), recvBuf(commDataSize);

    // Process both directions (low and high)
    for (int side = 0; side < 2; ++side) {
        int envelopeStart;
        if (side == 0) {  // Low side (west/south/down)
            envelopeStart = direction[0]*(nx - 2*envelopeWidth) + 
                           direction[1]*(ny - 2*envelopeWidth) + 
                           direction[2]*(nz - 2*envelopeWidth);
        } else {  // High side (east/north/up)
            envelopeStart = direction[0]*envelopeWidth + 
                           direction[1]*envelopeWidth + 
                           direction[2]*envelopeWidth;
        }

        packData(sendBuf, data_, direction, envelopeStart, dataSize, nx, ny, nz, envelopeWidth, currentStep);

        MPI_Sendrecv(sendBuf.data(), commDataSize, commDataType, recvRank[side], 0,
                     recvBuf.data(), commDataSize, commDataType, sendRank[side], 0,
                     world, &status);

        // Determine if we should unpack based on periodicity and processor position
        bool shouldUnpack = false;
        if (side == 0) {
            shouldUnpack = (procCoordinates[commDirection] != 0) || isPeriodic;
        } else {
            shouldUnpack = (procCoordinates[commDirection] != dimensions[commDirection]-1) || isPeriodic;
        }

        if (shouldUnpack) {
            int unpackStart;
            if (side == 0) {
                unpackStart = 0;
            } else {
                unpackStart = direction[0]*(nx - envelopeWidth) + 
                             direction[1]*(ny - envelopeWidth) + 
                             direction[2]*(nz - envelopeWidth);
            }
            unpackData(recvBuf, data_, direction, unpackStart, dataSize, nx, ny, nz, envelopeWidth, currentStep);
        }
    }
}

template<typename T> 
void LBMPSMMPI::packData(std::vector<T> &sendBuf, const std::vector<T> &data, const int direction[3],
                         int envelopeStart, int dataSize, int nx, int ny, int nz, int envelopeWidth, int currentStep) {
    const size_t baseIndex = static_cast<size_t>(nx)*ny*nz*dataSize*currentStep;
    size_t bufIndex = 0;
    
    // Optimized loop structure based on direction
    const int iStart = direction[0] ? envelopeStart : 0;
    const int iEnd = direction[0] ? (envelopeStart + envelopeWidth) : nx;
    const int jStart = direction[1] ? envelopeStart : 0;
    const int jEnd = direction[1] ? (envelopeStart + envelopeWidth) : ny;
    const int kStart = direction[2] ? envelopeStart : 0;
    const int kEnd = direction[2] ? (envelopeStart + envelopeWidth) : nz;
    
    for (int i = iStart; i < iEnd; ++i) {
        for (int j = jStart; j < jEnd; ++j) {
            for (int k = kStart; k < kEnd; ++k) {
                const size_t dataIndex = baseIndex + (static_cast<size_t>(i)*ny*nz + j*nz + k)*dataSize;
                for (int iq = 0; iq < dataSize; ++iq) {
                    sendBuf[bufIndex++] = data[dataIndex + iq];
                }
            }
        }
    }
}

template<typename T> 
void LBMPSMMPI::unpackData(const std::vector<T> &recvBuf, std::vector<T> &data, const int direction[3],
                           int envelopeStart, int dataSize, int nx, int ny, int nz, int envelopeWidth, int currentStep) {
    const size_t baseIndex = static_cast<size_t>(nx)*ny*nz*dataSize*currentStep;
    size_t bufIndex = 0;
    
    // Optimized loop structure based on direction
    const int iStart = direction[0] ? envelopeStart : 0;
    const int iEnd = direction[0] ? (envelopeStart + envelopeWidth) : nx;
    const int jStart = direction[1] ? envelopeStart : 0;
    const int jEnd = direction[1] ? (envelopeStart + envelopeWidth) : ny;
    const int kStart = direction[2] ? envelopeStart : 0;
    const int kEnd = direction[2] ? (envelopeStart + envelopeWidth) : nz;
    
    for (int i = iStart; i < iEnd; ++i) {
        for (int j = jStart; j < jEnd; ++j) {
            for (int k = kStart; k < kEnd; ++k) {
                const size_t dataIndex = baseIndex + (static_cast<size_t>(i)*ny*nz + j*nz + k)*dataSize;
                for (int iq = 0; iq < dataSize; ++iq) {
                    data[dataIndex + iq] = recvBuf[bufIndex++];
                }
            }
        }
    }
}

#endif