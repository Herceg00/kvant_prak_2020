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


complexd *generate_condition(int n) {
    long long unsigned qsize = 1LLU << n; // pow(2,n) for a condition-vector
    complexd *V = new complexd[qsize];
    double time_init = time(NULL);
    double module = 0;
    {
        unsigned int seed = omp_get_thread_num() * (unsigned) time_init;
        for (long long unsigned i = 0; i < qsize; i++) {
            V[i].real(rand_r(&seed) / (float) RAND_MAX - 0.5f);
            V[i].imag(rand_r(&seed) / (float) RAND_MAX - 0.5f);
            module += abs(V[i] * V[i]);
        }
        if (omp_get_thread_num() == 0) {
            module = sqrt(module);
        }
#pragma omp for schedule(static)
        for (long long unsigned j = 0; j < qsize; j++) {
            V[j] /= module;
        }
    }
    return V;
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
        MPI_Sendrecv(buf_zone, seg_size, MPI_DOUBLE_COMPLEX, rank_change, 0, recv_zone, seg_size, MPI_DOUBLE_COMPLEX, rank_change, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//        for (int i = 0; i < seg_size; i++) {
//            cout<<"RANK: "<<rank<<" index: "<<i<<" value: "<<recv_zone[i]<<endl;
//        }
        if (rank > rank_change) { //Got data somewhere from left
            for (int i = 0; i < seg_size; i++) {
                //cout<<"RANK: "<<rank<<" index: "<<i<<" value: "<<recv_zone[i]<<" "<<buf_zone[i]<<endl;

                recv_zone[i] = U[1][0] * recv_zone[i] + U[1][1] * buf_zone[i];
            }
        } else {
            for (int i = 0; i < seg_size; i++) {
                //cout<<"RANK: "<<rank<<" index: "<<i<<" value: "<<recv_zone[i]<<" "<<buf_zone[i]<<endl;
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
complexd *V = generate_condition(n);
     unsigned long long index = 1LLU << n;
complexd *W = new complexd[index];
//    int *V = new int[index];
//    int *W = new int[index];
//    for (int i = 0; i < index; i++) {
//        V[i] = i;
//    }
    complexd U[2][2];
    U[0][0] = 1 / sqrt(2);
    U[0][1] = 1 / sqrt(2);
    U[1][0] = 1 / sqrt(2);
    U[1][1] = -1 / sqrt(2);
//    int U[2][2];
//    U[0][0] = 1;
//    U[0][1] = 1;
//    U[1][0] = 1;
//    U[1][1] = 1;
    
    MPI_Init(&argc, &argv);
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    complexd *buf_zone = new complexd[index / size]; //TODO int div
    complexd *recv_buf = new complexd[index / size];
//    int *buf_zone = new int [index / size]; //TODO int div
//    int *recv_buf = new int [index / size];
    gettimeofday(&start, NULL);
    MPI_Scatter(V, index / size, MPI_DOUBLE_COMPLEX, buf_zone, index / size, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    OneQubitEvolution(buf_zone, U, n, k, recv_buf, rank, size);
    MPI_Gather(recv_buf, index / size, MPI_DOUBLE_COMPLEX, W, index / size, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    gettimeofday(&stop, NULL);
    
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
