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
    unsigned rank1_change = first_index ^(1u << (k - 1)); //ранк процесса, у которого отличается k
    unsigned rank2_change = first_index ^(1u << (l - 1)); //ранк процесса, у которого отличается l
    unsigned rank3_change = first_index ^((1u << (k - 1)) | (1u << (l - 1))); //отличается и k, и l

    rank1_change /= seg_size;
    rank2_change /= seg_size;
    rank3_change /= seg_size;
    std::cout<<"process of rank "<<rank<<" has "<<rank1_change<<" , "<<rank2_change<<" , "<<rank3_change<<endl;


    if (rank ==
        rank3_change) { //The case when we don't need to change data - ранк противоположного процесса равен собственному
#pragma omp parallel shared(buf0, U)
        {
#pragma omp for schedule(static)
            for (std::size_t i = 0; i < seg_size; i++) {
                unsigned k_bit = (i >> (k-1)) & 1u;
                unsigned l_bit = (i >> (l-1)) & 1u;
                //cout<<"RANK: "<<rank<<" element "<<i<<" bits: "<<k_bit<<" "<<l_bit<<endl;
                if ((!k_bit) && (!l_bit)) //00
                {
                    cout<<i<<" is for 00";
                    size_t a = i | (1u << (k - 1));//01
                    size_t b = i | (1u << (l - 1));//10
                    size_t c = b | (1u << (k - 1));//11
                    buf0[i] = U[0][0] * buf0[i] + U[1][0] * buf0[a] + U[2][0] * buf0[b] + U[3][0] * buf0[c];
                }
                if ((!k_bit) && (l_bit)) //01
                {
                    cout<<i<<" is for 01";
                    size_t a = i ^(1u << (l - 1));//00
                    size_t b = a | (1u << (k - 1));//10
                    size_t c = i | (1u << (k - 1));//11
                    buf0[i] = U[0][1] * buf0[a] + U[1][1] * buf0[i] + U[2][1] * buf0[b] + U[3][1] * buf0[c];
                }
                if ((k_bit) && (!l_bit)) //10
                {
                    cout<<i<<" is for 10";
                    size_t a = i ^(1u << (k - 1));//00
                    size_t b = a | (1u << (l - 1));//01
                    size_t c = i | (1u << (l - 1));//11
                    buf0[i] = U[0][2] * buf0[a] + U[1][2] * buf0[b] + U[2][2] * buf0[i] + U[3][2] * buf0[c];
                }
                if ((k_bit) && (l_bit)) //11
                {
                    cout<<i<<" is for 11";
                    size_t a = i ^((1u << (k - 1)) | (1u << (l - 1)));//00
                    size_t b = a | (1u << (l - 1));//01
                    size_t c = a | (1u << (k - 1));//10
                    buf0[i] = U[0][3] * buf0[a] + U[1][3] * buf0[b] + U[2][3] * buf0[c] + U[3][3] * buf0[i];
                }
            }
        }

    } else if ((rank == rank1_change) ||
               (rank == rank2_change)) { //The case when one array should be get from other process
        if (k > l) { //Обмениваемся с процессом, соседним по l (l совпадает)
            MPI_Sendrecv(buf0, seg_size, MPI_DOUBLE_COMPLEX, rank1_change, 0, buf1, seg_size, MPI_DOUBLE_COMPLEX,
                         rank1_change, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else { //Обмениваемся с процессом, соседним по k (k совпадает)
            MPI_Sendrecv(buf0, seg_size, MPI_DOUBLE_COMPLEX, rank2_change, 0, buf1, seg_size, MPI_DOUBLE_COMPLEX,
                         rank2_change, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
#pragma omp parallel shared(buf0, buf1, U)
        {
#pragma omp for schedule(static)
            for (std::size_t i = 0; i < seg_size; i++) {
                char l_bit = (i >> (l-1)) & 1u;
                char k_bit = (i >> (k-1)) & 1u;
                if ((!k_bit) && (!l_bit)) { //00
                    if (k > l) {
                        char l_pair = i | (1 << (l - 1));
                        buf0[i] =
                                U[0][0] * buf0[i] + U[1][0] * buf0[l_pair] + U[2][0] * buf1[i] + U[3][0] * buf1[l_pair];
                    } else {
                        char k_pair = i | (1 << (k - 1));
                        buf0[i] =
                                U[0][0] * buf0[i] + U[1][0] * buf1[i] + U[2][0] * buf0[k_pair] + U[3][0] * buf1[k_pair];
                    }
                } else if ((!k_bit) && (l_bit)) { //01
                    if (k > l) {
                        char l_pair = i ^(1 << (l - 1));
                        buf0[i] =
                                U[0][1] * buf0[l_pair] + U[1][1] * buf0[i] + U[2][1] * buf1[l_pair] + U[3][1] * buf1[i];
                    } else {
                        char k_pair = i | (1 << (k - 1));
                        buf0[i] =
                                U[0][1] * buf1[i] + U[1][1] * buf0[i] + U[2][1] * buf1[k_pair] + U[3][1] * buf0[k_pair];
                    }
                } else if ((k_bit) && (!l_bit)) { //10
                    if (k > l) {
                        char l_pair = i | (1 << (l - 1));
                        buf0[i] =
                                U[0][2] * buf1[i] + U[1][2] * buf1[l_pair] + U[2][2] * buf0[i] + U[3][2] * buf0[l_pair];
                    } else {
                        char k_pair = i ^(1 << (k - 1));
                        buf0[i] =
                                U[0][2] * buf0[k_pair] + U[1][2] * buf1[k_pair] + U[2][2] * buf0[i] + U[3][2] * buf1[i];
                    }
                } else { //11
                    if (k > l) {
                        char l_pair = i ^(1 << (l - 1));
                        buf0[i] =
                                U[0][3] * buf1[l_pair] + U[1][3] * buf1[i] + U[2][3] * buf0[l_pair] + U[3][3] * buf0[i];
                    } else {
                        char k_pair = i ^(1 << (k - 1));
                        buf0[i] =
                                U[0][3] * buf1[k_pair] + U[1][3] * buf0[k_pair] + U[2][3] * buf1[i] + U[3][3] * buf0[i];
                    }
                }
            }
        }

    } else { //The case when we should gather data from all processes

//buf 1 has !k_local and l_local - происходит обмен c процессом, соседним по l (имеет различный k)
        MPI_Sendrecv(buf0, seg_size, MPI_DOUBLE_COMPLEX, rank1_change, 0, buf1, seg_size, //close process by k
                     MPI_DOUBLE_COMPLEX,
                     rank1_change, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//buf 2 has k_local and !l_local - происходит обмен с процессом, соседним по k (имеет различный l)
        MPI_Sendrecv(buf0, seg_size, MPI_DOUBLE_COMPLEX, rank2_change, 0, buf2, seg_size, //close process by l
                     MPI_DOUBLE_COMPLEX,
                     rank2_change, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//buf 3 has !k_local and !l_local - происходит обмен с противоположным процессом (имеет различные k и l)
        MPI_Sendrecv(buf0, seg_size, MPI_DOUBLE_COMPLEX, rank3_change, 0, buf3, seg_size, //opposite process
                     MPI_DOUBLE_COMPLEX,
                     rank3_change, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#pragma omp parallel shared(buf0,buf1,buf2, U)
        {
#pragma omp for schedule(static)
            for (std::size_t i = 0; i < seg_size; i++) {
                char k_bit = (i >> (k-1)) & 1u;
                char l_bit = (i >> (l-1)) & 1u;
                if ((!k_bit) && (!l_bit)) { //00
                    buf0[i] = U[0][0] * buf0[i] + U[1][0] * buf2[i] + U[2][0] * buf1[i] + U[3][0] * buf3[i];
                }
                if ((!k_bit) && (!l_bit)) { //01
                    buf0[i] = U[0][1] * buf2[i] + U[1][1] * buf0[i] + U[2][1] * buf3[i] + U[3][1] * buf1[i];
                }
                if ((!k_bit) && (!l_bit)) { //10
                    buf0[i] = U[0][2] * buf1[i] + U[1][2] * buf3[i] + U[2][2] * buf0[i] + U[3][2] * buf2[i];
                }
                if ((!k_bit) && (!l_bit)) { //11
                    buf0[i] = U[0][3] * buf3[i] + U[1][3] * buf1[i] + U[2][3] * buf2[i] + U[3][3] * buf0[i];
                }
            }
        }
    }
}

void NOT(unsigned k, complexd *buf0, int rank, int size, unsigned n) {
    complexd *buf1;
    unsigned N = 1u << n;
    unsigned seg_size = N / size;
    if ((1u << k > seg_size)) { //нужен обмен
        buf1 = new complexd[seg_size];
    }
    complexd U[2][2];
    U[0][0] = 0;
    U[1][1] = 0;
    U[0][1] = 1;
    U[1][0] = 1;
    OneQubitEvolution(buf0, U, n, k, buf1, rank, size);
}

void ROT(unsigned k, complexd *buf0, int rank, int size, unsigned n, double thetta) {
    complexd *buf1;
    unsigned N = 1u << n;
    unsigned seg_size = N / size;
    if ((1u << k > seg_size)) { //нужен обмен
        buf1 = new complexd[seg_size];
    }
    complexd a(0.0, 1.0);
    complexd U[2][2];
    U[0][0] = 1;
    U[1][1] = exp(a.imag() * thetta);
    U[0][1] = 0;
    U[1][0] = 0;
    OneQubitEvolution(buf0, U, n, k, buf1, rank, size);
}

void CNOT(unsigned k, unsigned l, complexd *buf0, int rank, int size, unsigned n) {
    complexd *buf1, *buf2, *buf3;
    unsigned N = 1u << n;
    unsigned seg_size = N / size;
    if ((1u << l > seg_size) && (1u << k > seg_size)) { //нужно 3 обмена
        buf1 = new complexd[seg_size];
        buf2 = new complexd[seg_size];
        buf3 = new complexd[seg_size];
    } else if ((1u << l > seg_size) || (1u << k > seg_size)) { //нужен 1 обмен
        buf1 = new complexd[seg_size];
    }
    complexd U[4][4];
    for (short i = 0; i < 4; i++) {
        for (short j = 0; i < 4; i++) {
            U[i][j] = 0;
        }
    }
    U[0][0] = 1;
    U[1][1] = 1;
    U[2][3] = 1;
    U[3][2] = 1;
    TwoQubitEvolution(buf0, buf1, buf2, buf3, U, n, k, l, rank, size);
}

void CROT(unsigned k, unsigned l, complexd *buf0, int rank, int size, unsigned n, double thetta) {
    complexd *buf1, *buf2, *buf3;
    unsigned N = 1u << n;
    unsigned seg_size = N / size;
    if ((1u << l > seg_size) && (1u << k > seg_size)) { //нужно 3 обмена
        buf1 = new complexd[seg_size];
        buf2 = new complexd[seg_size];
        buf3 = new complexd[seg_size];
    } else if ((1u << l > seg_size) || (1u << k > seg_size)) { //нужен 1 обмен
        buf1 = new complexd[seg_size];
    }
    complexd U[4][4];
    for (short i = 0; i < 4; i++) {
        for (short j = 0; i < 4; i++) {
            U[i][j] = 0;
        }
    }
    complexd a(0.0, 1.0);
    U[0][0] = 1;
    U[1][1] = 1;
    U[2][3] = exp(a.imag() * thetta);
    U[3][2] = 1;
    double begin = MPI_Wtime();
    TwoQubitEvolution(buf0, buf1, buf2, buf3, U, n, k, l, rank, size);
    double end = MPI_Wtime();
    std::cout << "The process took " << end - begin << " seconds to run." << std::endl;
}

std::size_t difference(complexd *ideal, complexd *result, unsigned long long seg_size, int rank) {
    std::size_t error_position = 0;
#pragma omp parallel shared(ideal, error_position)
    {
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < seg_size; i++) {
            if (ideal[i] != result[i]) {
                error_position = i + seg_size * rank;
            }
        }
    }
    return error_position; //Last error position will be fixed
}

//blackbox is tuned for CROT gate
std::size_t blackbox(complexd *ideal, complexd U[4][4], unsigned n, unsigned k, unsigned l, unsigned long long seg_size, int rank, int size) {
    complexd *buf0, *buf1, *buf2, *buf3;
    if ((1u << l > seg_size) && (1u << k > seg_size)) { //нужно 3 обмена
        buf1 = new complexd[seg_size];
        buf2 = new complexd[seg_size];
        buf3 = new complexd[seg_size];
    } else if ((1u << l > seg_size) || (1u << k > seg_size)) { //нужен 1 обмен
        buf1 = new complexd[seg_size];
    }
#pragma omp parallel shared(ideal, buf0)
    {
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < seg_size; i++) {
            buf0[i] = ideal[i];
        }
    }
    TwoQubitEvolution(buf0, buf1, buf2, buf3, U, n, k, l, rank, size);
    complexd b(1.0, 0.0);
    U[3][3] = b / U[3][3];
    TwoQubitEvolution(buf0, buf1, buf2, buf3, U, n, k, l, rank, size);
    size_t position = difference(ideal, buf0, seg_size, rank);
    return position;
}


#endif //KVANT_MPI_GATES_LIBRARY_H
