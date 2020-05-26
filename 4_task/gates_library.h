#ifndef KVANT_MPI_GATES_LIBRARY_H
#define KVANT_MPI_GATES_LIBRARY_H

#include "mpi/mpi.h"
#include <iostream>
#include <complex>

using namespace std;
typedef complex<double> complexd;

void
OneQubitEvolution(complexd *buf_zone, complexd U[2][2], unsigned int n, unsigned int k, complexd *recv_zone, int rank,
                  int size) {
    unsigned N = 1u << n;
    unsigned seg_size = N / size;
    unsigned first_index = rank * seg_size;
    unsigned rank_change = first_index ^(1u << (k - 1));
    rank_change /= seg_size;

    //printf("RANK %d has a change-neighbor %d\n", rank, rank_change);
    if (rank != rank_change) {
        MPI_Sendrecv(buf_zone, seg_size, MPI_DOUBLE_COMPLEX, rank_change, 0, recv_zone, seg_size, MPI_DOUBLE_COMPLEX,
                     rank_change, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank > rank_change) { //Got data somewhere from left
#pragma omp parallel shared(recv_zone, buf_zone, U)
            {
#pragma omp for schedule(static)
                for (int i = 0; i < seg_size; i++) {
                    recv_zone[i] = U[1][0] * recv_zone[i] + U[1][1] * buf_zone[i];
                }
            }
        } else {
#pragma omp parallel shared(recv_zone, buf_zone, U)
            {
#pragma omp for schedule(static)
                for (int i = 0; i < seg_size; i++) {
                    recv_zone[i] = U[0][0] * buf_zone[i] + U[0][1] * recv_zone[i];
                }
            }
        }
    } else {
        unsigned shift = (int) log2(seg_size) - k;
        unsigned pow = 1u << (shift);
#pragma omp parallel shared(recv_zone, buf_zone, U)
        {
#pragma omp for schedule(static)
            for (std::size_t i = 0; i < seg_size; i++) {
                unsigned i0 = i & ~pow;
                unsigned i1 = i | pow;
                unsigned iq = (i & pow) >> shift;
                recv_zone[i] = U[iq][0] * buf_zone[i0] + U[iq][1] * buf_zone[i1];
            }
        }
    }
}


//k is a control qubit, l is a controlled qubit, buf{i} - arrays to be passed as vectors to matrix multiplication
void TwoQubitEvolution(complexd *buf0, complexd *buf1, complexd *buf2, complexd *buf3, complexd U[4][4], unsigned int n,
                       unsigned int k, unsigned int l, int rank,
                       int size) {
    unsigned N = 1u << n;
    unsigned seg_size = N / size;
    unsigned first_index = rank * seg_size;
    unsigned rank1_change = first_index ^(1u << (k - 1));
    unsigned rank2_change = first_index ^(1u << (l - 1));
    unsigned rank3_change = first_index ^((1u << (k - 1)) | (1u << (l - 1)));
    rank1_change /= seg_size;
    rank2_change /= seg_size;
    rank3_change /= seg_size;
    //The case when we don't need to change data
    if (rank == rank3_change) {
#pragma omp parallel shared(recv_zone, buf_zone, U)
        {
#pragma omp for schedule(static)
            for (std::size_t i = 0; i < seg_size; i++) {
                char k_bit = (i>>k)&1;
                char l_bit = (i>>l)&1;
                if ((!k_bit)&&(!l_bit)) //00
                {
                    size_t a = i|(1<<(k-1));//01
                    size_t b = i|(1<<(l-1));//10
                    size_t c = b|(1<<(k-1));//11
                    buf0[i] = U[0][0]*buf0[i] + U[1][0]*buf0[a] + U[2][0]*buf0[b] + U[3][0]*buf0[c];
                }
                if ((!k_bit)&&(l_bit)) //01
                {
                    size_t a = i^(1<<(l-1));//00
                    size_t b = a|(1<<(k-1));//10
                    size_t c = i|(1<<(k-1));//11
                    buf0[i] = U[0][1]*buf0[a] + U[1][1]*buf0[i] + U[2][1]*buf0[b] + U[3][1]*buf0[c];
                }
                if ((k_bit)&&(!l_bit)) //10
                {
                    size_t a = i^(1<<(k-1));//00
                    size_t b = a|(1<<(l-1));//01
                    size_t c = i|(1<<(l-1));//11
                    buf0[i] = U[0][2]*buf0[a] + U[1][2]*buf0[b] + U[2][2]*buf0[i] + U[3][2]*buf0[c];
                }
                if ((k_bit)&&(l_bit)) //11
                {
                    size_t a = i^((1<<(k-1))|(1<<(l-1)));//00
                    size_t b = a|(1<<(l-1));//01
                    size_t c = a|(1<<(k-1));//10
                    buf0[i] = U[0][3]*buf0[a] + U[1][3]*buf0[b] + U[2][3]*buf0[c] + U[3][3]*buf0[i];
                }
            }
        }
        //The case when one array should be get from other process
    } else if ((rank == rank1_change) || (rank == rank2_change)) {
        if(k>l) {
            MPI_Sendrecv(buf0, seg_size, MPI_DOUBLE_COMPLEX, rank1_change, 0, buf1, seg_size,
                         MPI_DOUBLE_COMPLEX,
                         rank1_change, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            short k_local = rank*seg_size>>k; //k is common for all elements in a process chunk
            if (!k_local){
#pragma omp parallel shared(recv_zone, buf_zone, U)
                {
#pragma omp for schedule(static)
                    for (std::size_t i = 0; i < seg_size; i++) {
                        char l_bit = (i>>l)&1;
                        if(!l_bit){ //00
                            char l_pair = i|(1<<(l-1));
                            buf0[i] = U[0][0]*buf0[i] + U[1][0]*buf0[l_pair] + U[2][0]*buf1[i] + U[3][0]*buf1[l_pair];
                        } else{ //01
                            char l_pair = i^(1<<(l-1));
                            buf0[i] = U[0][1]*buf0[l_pair] + U[1][1]*buf0[i] + U[2][1]*buf1[l_pair] + U[3][1]*buf1[i];
                        }
                    }
                }
            } else {
#pragma omp parallel shared(recv_zone, buf_zone, U)
                {
#pragma omp for schedule(static)
                    for (std::size_t i = 0; i < seg_size; i++) {
                        char l_bit = (i>>l)&1;
                        if(!l_bit){ //10
                            char l_pair = i|(1<<(l-1));
                            buf0[i] = U[0][2]*buf1[i] + U[1][2]*buf1[l_pair] + U[2][2]*buf0[i] + U[3][2]*buf0[l_pair];
                        } else{ //11
                            char l_pair = i^(1<<(l-1));
                            buf0[i] = U[0][3]*buf1[l_pair] + U[1][3]*buf1[i] + U[2][3]*buf0[l_pair] + U[3][3]*buf0[i];
                        }
                    }
                }

            }
        } else {
            MPI_Sendrecv(buf0, seg_size, MPI_DOUBLE_COMPLEX, rank2_change, 0, buf1, seg_size,
                         MPI_DOUBLE_COMPLEX,
                         rank2_change, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            short l_local = rank*seg_size>>l; //l is common for all elements in a process chunk
            if (!l_local){ //*0
#pragma omp parallel shared(recv_zone, buf_zone, U)
                {
#pragma omp for schedule(static)
                    for (std::size_t i = 0; i < seg_size; i++) {
                        char k_bit = (i>>k)&1;
                        if(!k_bit){ //00
                            char k_pair = i|(1<<(k-1));
                            buf0[i] = U[0][0]*buf0[i] + U[1][0]*buf1[i] + U[2][0]*buf0[k_pair] + U[3][0]*buf1[k_pair];
                        } else { //10
                            char k_pair = i^(1<<(k-1));
                            buf0[i] = U[0][2]*buf0[k_pair] + U[1][2]*buf1[k_pair] + U[2][2]*buf0[i] + U[3][2]*buf1[i];
                        }
                    }
                }
            } else { //*1
#pragma omp parallel shared(recv_zone, buf_zone, U)
                {
#pragma omp for schedule(static)
                    for (std::size_t i = 0; i < seg_size; i++) {
                        char k_bit = (i>>k)&1;
                        if(!k_bit){ //01
                            char k_pair = i|(1<<(k-1));
                            buf0[i] = U[0][1]*buf1[i] + U[1][1]*buf0[i] + U[2][1]*buf1[k_pair] + U[3][1]*buf0[k_pair];
                        } else { //11
                            char k_pair = i^(1<<(k-1));
                            buf0[i] = U[0][3]*buf1[k_pair] + U[1][3]*buf0[k_pair] + U[2][3]*buf1[i] + U[3][3]*buf1[i];
                        }
                    }
                }
            }
        }
        //The case when we should gather data from all processes
    } else {
        short l_local = rank*seg_size>>l; //k and l are identical for this process
        short k_local = rank*seg_size>>k;
        //buf 1 has !k_local and l_local
        MPI_Sendrecv(buf0, seg_size, MPI_DOUBLE_COMPLEX, rank1_change, 0, buf1, seg_size, //close process by k
                     MPI_DOUBLE_COMPLEX,
                     rank1_change, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //buf 2 has k_local and !l_local
        MPI_Sendrecv(buf0, seg_size, MPI_DOUBLE_COMPLEX, rank2_change, 0, buf2, seg_size, //close process by l
                     MPI_DOUBLE_COMPLEX,
                     rank2_change, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //buf 3 has !k_local and !l_local
        MPI_Sendrecv(buf0, seg_size, MPI_DOUBLE_COMPLEX, rank3_change, 0, buf3, seg_size, //opposite process
                     MPI_DOUBLE_COMPLEX,
                     rank3_change, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#pragma omp parallel shared(recv_zone, buf_zone, U)
        {
#pragma omp for schedule(static)
            for (std::size_t i = 0; i < seg_size; i++) {
                char k_bit = (i>>k)&1;
                char l_bit = (i>>l)&1;
                if((!k_bit)&&(!l_bit)){ //00
                    buf0[i] = U[0][0]*buf0[i] + U[1][0]*buf2[i] + U[2][0]*buf1[i] + U[3][0]*buf3[i];
                }
                if((!k_bit)&&(!l_bit)){ //01
                    buf0[i] = U[0][1]*buf2[i] + U[1][1]*buf0[i] + U[2][1]*buf3[i] + U[3][1]*buf1[i];
                }
                if((!k_bit)&&(!l_bit)){ //10
                    buf0[i] = U[0][2]*buf1[i] + U[1][2]*buf3[i] + U[2][2]*buf0[i] + U[3][2]*buf2[i];
                }
                if((!k_bit)&&(!l_bit)){ //11
                    buf0[i] = U[0][3]*buf3[i] + U[1][3]*buf1[i] + U[2][3]*buf2[i] + U[3][3]*buf0[i];
                }
            }
        }
    }
}


#endif //KVANT_MPI_GATES_LIBRARY_H
