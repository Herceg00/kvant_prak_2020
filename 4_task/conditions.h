#ifndef KVANT_MPI_CONDITIONS_H
#define KVANT_MPI_CONDITIONS_H
#include "mpi/mpi.h"
#include <iostream>
#include <complex>

using namespace std;
typedef complex<double> complexd;


complexd *generate_condition(unsigned long long seg_size, int rank) {
    auto *A = new complexd[seg_size];
    double sqr = 0, module;
    unsigned int seed = time(nullptr) + rank;
#pragma omp parallel shared(A) reduction(+: sqr)
    {
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < seg_size; i++) {
            A[i].real((rand_r(&seed) / (float) RAND_MAX) - 0.5f);
            A[i].imag((rand_r(&seed) / (float) RAND_MAX) - 0.5f);
            sqr += abs(A[i] * A[i]);
        }
    }
    MPI_Reduce(&sqr, &module, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        module = sqrt(module);
    }
    MPI_Bcast(&module, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#pragma omp parallel shared(A, module)
    {
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < seg_size; i++)
            A[i] /= module;
    }

    return A;
}

complexd *read(char *f, unsigned int *n, int rank, int size) {
    MPI_File file;
    if (MPI_File_open(MPI_COMM_WORLD, f, MPI_MODE_RDONLY, MPI_INFO_NULL,
                      &file)) {
        if (!rank)
            printf("Error opening file %s\n", f);
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
    }
    if (!rank)
        MPI_File_read(file, n, 1, MPI_INT, MPI_STATUS_IGNORE);
    MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    unsigned long long index = 1LLU << *n;
    cout << size << *n << endl;
    unsigned seg_size = index / size;
    auto *A = new complexd[seg_size];

    double d[2];
    MPI_File_seek(file, sizeof(int) + 2 * seg_size * rank * sizeof(double),
                  MPI_SEEK_SET);
    for (std::size_t i = 0; i < seg_size; ++i) {
        MPI_File_read(file, &d, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
        A[i].real(d[0]);
        A[i].imag(d[1]);
    }
    MPI_File_close(&file);
    return A;
}


void write(char *f, complexd *B, int n, int rank, int size) {
    MPI_File file;
    if (MPI_File_open(MPI_COMM_WORLD, f, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                      MPI_INFO_NULL, &file)) {
        if (!rank)
            printf("Error opening file %s\n", f);
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
    }
    if (rank == 0) {
        MPI_File_write(file, &n, 1, MPI_INT, MPI_STATUS_IGNORE);

    }
    unsigned long long index = 1LLU << n;
    unsigned seg_size = index / size;
    double d[2];
    MPI_File_seek(file, sizeof(int) + 2 * seg_size * rank * sizeof(double), MPI_SEEK_SET);
    for (std::size_t i = 0; i < seg_size; ++i) {
        d[0] = B[i].real();
        d[1] = B[i].imag();
        MPI_File_write(file, &d, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&file);
}

#endif //KVANT_MPI_CONDITIONS_H
