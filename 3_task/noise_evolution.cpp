#include <iostream>
#include <complex>
#include "omp.h"
#include <cmath>
#include "time.h"
#include "sys/time.h"
#include <stdio.h>
#include <stdlib.h>
#include "mpi/mpi.h"

#define eps 0.01
using namespace std;

double normal_dis_gen() //generate a value
{
    double S = 0.;
    for (int i = 0; i < 12; ++i) { S += (double) rand() / RAND_MAX; }
    return S - 6.0;
}

typedef complex<double> complexd;

void make_noise(complexd U[2][2], complexd V[2][2], bool noise) {

    if (noise) {
        double thetta = normal_dis_gen() * eps;
        V[0][0] = U[0][0] * cos(thetta) - U[0][1] * sin(thetta);
        V[0][1] = U[0][0] * sin(thetta) + U[0][1] * cos(thetta);
        V[1][0] = U[1][0] * cos(thetta) - U[1][1] * sin(thetta);
        V[1][1] = U[1][0] * sin(thetta) + U[1][1] * cos(thetta);
    } else {
        V[0][0] = U[0][0];
        V[0][1] = U[0][1];
        V[1][0] = U[1][0];
        V[1][1] = U[1][1];

    }
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


void
OneQubitEvolution(complexd *buf_zone, complexd U[2][2], unsigned int n, unsigned int k, complexd *recv_zone, int rank,
                  int size, bool noise) {
    unsigned N = 1u << n;
    unsigned seg_size = N / size;
    unsigned first_index = rank * seg_size;
    unsigned rank_change = first_index ^(1u << (k - 1));
    rank_change /= seg_size;

    printf("RANK %d has a change-neighbor %d\n", rank, rank_change);
    complexd V[2][2];
    make_noise(U, V, noise);

    if (rank != rank_change) {
        MPI_Sendrecv(buf_zone, seg_size, MPI_DOUBLE_COMPLEX, rank_change, 0, recv_zone, seg_size, MPI_DOUBLE_COMPLEX,
                     rank_change, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank > rank_change) { //Got data somewhere from left
#pragma omp parallel shared(recv_zone, buf_zone, V)
            {
#pragma omp for schedule(static)
                for (int i = 0; i < seg_size; i++) {
                    recv_zone[i] = V[1][0] * recv_zone[i] + V[1][1] * buf_zone[i];
                }
            }
        } else {
#pragma omp parallel shared(recv_zone, buf_zone, V)
            {
#pragma omp for schedule(static)
                for (int i = 0; i < seg_size; i++) {
                    recv_zone[i] = V[0][0] * buf_zone[i] + V[0][1] * recv_zone[i];
                }
            }
        }
    } else {
        unsigned shift = (int) log2(seg_size) - k;
        unsigned pow = 1u << (shift);
#pragma omp parallel shared(recv_zone, buf_zone, V)
        {
#pragma omp for schedule(static)
            for (std::size_t i = 0; i < seg_size; i++) {
                unsigned i0 = i & ~pow;
                unsigned i1 = i | pow;
                unsigned iq = (i & pow) >> shift;
                recv_zone[i] = V[iq][0] * buf_zone[i0] + V[iq][1] * buf_zone[i1];
            }
        }
    }
}

std::size_t difference(complexd *ideal, complexd *count, unsigned long long seg_size, int rank) {
    std::size_t error_position = 0;
#pragma omp parallel shared(ideal, count, error_position)
    {
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < seg_size; i++) {
            if (ideal[i] != count[i]) {
                error_position = i + seg_size * rank;
            }
        }
    }
    return error_position; //Last error position will be fixed
}

double get_distance(complexd *ideal, complexd *noise, int rank, unsigned long long seg_size) {
    double sqr = 0;
    double module;
#pragma omp parallel shared(ideal, noise) reduction(+: sqr)
    {
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < seg_size; i++) {
            sqr += abs(ideal[i] * noise[i]) * abs(ideal[i] * noise[i]);
        }
    }
    MPI_Reduce(&sqr, &module, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        return module;
    } else {
        return 0;
    }
}


int main(int argc, char **argv) {
    bool file_read = false;
    bool test_flag = false;
    char *input, *output, *test_file;
    unsigned k, n;
    for (int i = 1; i < argc; i++) {
        string option(argv[i]);

        if (option.compare("n") == 0) {
            n = atoi(argv[++i]);
        }

        if ((option.compare("k") == 0)) {
            k = atoi(argv[++i]);
        }

        if ((option.compare("file_read") == 0)) {
            input = argv[++i];
            file_read = true;
        }
        if ((option.compare("file_write") == 0)) {
            output = argv[++i];
        }
        if ((option.compare("test") == 0)) {
            test_flag = true;
        }
        if ((option.compare("file_test") == 0)) {
            test_file = argv[++i];
        }
    }

    MPI_Init(&argc, &argv);
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    unsigned long long index = 1LLU << n;
    unsigned long long seg_size = index / size;
    complexd *V;
    if (!file_read) {
        V = generate_condition(seg_size, rank);
    } else {
        V = read(input, &n, rank, size);
    }

    complexd U[2][2];
    U[0][0] = 1 / sqrt(2);
    U[0][1] = 1 / sqrt(2);
    U[1][0] = 1 / sqrt(2);
    U[1][1] = -1 / sqrt(2);


    auto *recv_buf1 = new complexd[seg_size];
    auto *recv_buf2 = new complexd[seg_size];
    //double thetta = normal_dis_gen() * eps;

    double begin = MPI_Wtime();

    //simple evolution
    for (int qubit = 1; qubit < n + 1; qubit++) {
        OneQubitEvolution(V, U, n, k, recv_buf1, rank, size, false);
    }


    //noise evolution
    for (int qubit = 1; qubit < n + 1; qubit++) {
        OneQubitEvolution(V, U, n, k, recv_buf2, rank, size, true);
    }
    double end = MPI_Wtime();

    std::cout << "The process took " << end - begin << " seconds to run." << std::endl;
    if (test_flag) {
        complexd *test_vector = read(test_file, &n, rank, size);
        if (std::size_t pos = difference(test_vector, V, seg_size, rank) == 0) {
            cout << "Correct for rank " << rank;
        } else {
            cout << "Error in " << pos << " position on segment number " << rank;
        }
    } else {
        double distance = get_distance(recv_buf1, recv_buf2, rank, seg_size);
        if (!rank) {                                                  //only root has non-NULL value
            cout << "distance is equal " << distance << endl;
        }
    }
    MPI_Finalize();
    delete[] V;

}