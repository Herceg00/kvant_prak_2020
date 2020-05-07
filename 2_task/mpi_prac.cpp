#include <iostream>
#include <complex>
#include <assert.h>
#include "omp.h"
#include <cmath>
#include "time.h"
#include "sys/time.h"
#include <stdio.h>
#include <stdlib.h>
#include "mpi/mpi.h"

using namespace std;


typedef complex<double> complexd;

complexd* generate_condition(int seg_size, int rank)
{
    complexd *A = new complexd[seg_size];
    double sqr = 0, module;
    unsigned int seed = time(nullptr) + rank;
    for (std::size_t i = 0; i < seg_size; i++) {
        A[i].real((rand_r(&seed) / (float) RAND_MAX) - 0.5f);
        A[i].imag((rand_r(&seed) / (float) RAND_MAX) - 0.5f);
        sqr += abs(A[i] * A[i]);
    }
    MPI_Reduce(&sqr, &module, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        module = sqrt(module);
    }
    MPI_Bcast(&module, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (std::size_t i = 0; i < seg_size; i++)
        A[i] /= module;
    return A;
}

void
OneQubitEvolution(complexd *buf_zone, complexd U[2][2], unsigned int n, unsigned int k, complexd *recv_zone, int rank,
                  int size) {

    unsigned N = 1u << n;
    unsigned seg_size = N / size;
    unsigned first_index = rank * seg_size;
    unsigned rank_change = first_index ^(1u << (k - 1));
    rank_change /= seg_size;

    printf("RANK %d has a change-neighbor %d\n", rank, rank_change);

    if (rank != rank_change) {
        MPI_Sendrecv(buf_zone, seg_size, MPI_DOUBLE_COMPLEX, rank_change, 0, recv_zone, seg_size, MPI_DOUBLE_COMPLEX,
                     rank_change, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank > rank_change) { //Got data somewhere from left
            for (int i = 0; i < seg_size; i++) {
                recv_zone[i] = U[1][0] * recv_zone[i] + U[1][1] * buf_zone[i];
            }
        } else {
            for (int i = 0; i < seg_size; i++) {
                recv_zone[i] = U[0][0] * buf_zone[i] + U[0][1] * recv_zone[i];
            }
        }
    } else {
        int shift = seg_size - k;
        int pow = 1 << (shift);
#pragma omp for schedule(static)
        for (int i = 0; i < seg_size; i++) {
            int i0 = i & ~pow;
            int i1 = i | pow;
            int iq = (i & pow) >> shift;
            recv_zone[i] = U[iq][0] * buf_zone[i0] + U[iq][1] * buf_zone[i1];
        }
    }
}


int main(int argc, char **argv) {
    if (argc < 3) {
        cout << "not enough arguments";
        return 1;
    }
    int n = atoi(argv[1]); // number of qubits
    int k = atoi(argv[2]); // operated qubit number
    assert(n >= k);
    struct timeval start, stop;

    unsigned long long index = 1LLU << n;
    complexd *W = new complexd[index];
    complexd U[2][2];
    U[0][0] = 1 / sqrt(2);
    U[0][1] = 1 / sqrt(2);
    U[1][0] = 1 / sqrt(2);
    U[1][1] = -1 / sqrt(2);

    MPI_Init(&argc, &argv);
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int seg_size = index/size;
    complexd *recv_buf = new complexd[seg_size];
    complexd *V = generate_condition(seg_size, rank);
    gettimeofday(&start, nullptr);
    OneQubitEvolution(V, U, n, k, recv_buf, rank, size);
    MPI_Gather(recv_buf, seg_size, MPI_DOUBLE_COMPLEX, W, seg_size, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    gettimeofday(&stop, nullptr);
    if (rank == 0) {
        for (int i = 0; i < index; i++) {
            cout << W[i] << endl;
        }
    }
    printf(" LIFETIME:%lf s ",
           (float) ((stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec) / 1000000);
    delete[] V;
    delete[] W;
    MPI_Finalize();
}
