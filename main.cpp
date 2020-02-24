#include <iostream>
#include <complex>
#include <assert.h>
#include "omp.h"
#include <cmath>
#include "time.h"
#include "sys/time.h"
#include <stdio.h>
#include <stdlib.h>

using namespace std;


typedef complex<double> complexd;




complexd* generate_condition(int n){
    long long unsigned qsize = 1LLU<<n; // pow(2,n) for a condition-vector
    complexd *V = new complexd[qsize];
    double time_init = time(NULL);
    double module =0;
#pragma omp parallel shared(V,time_init) reduction(+: module)
    {
        unsigned int seed = omp_get_thread_num() * (unsigned)time_init;
#pragma omp for schedule(static)
        for (long long unsigned i = 0;  i< qsize ; i++) {
            V[i].real(rand_r(&seed)/(float)RAND_MAX - 0.5f);
            V[i].imag(rand_r(&seed)/(float)RAND_MAX - 0.5f);
            module += abs(V[i] * V[i]);
        }
        if(omp_get_thread_num() == 0){
            module = sqrt(module);
        }
#pragma omp for schedule(static)
        for (long long unsigned j = 0; j < qsize; j++) {
            V[j] /= module;
        }
    }
    return V;
}



void OneQubitEvolution(complexd* V,complexd* W,complexd U[2][2],int n, int k){
    int shift = n-k;
    int pow=1<<(shift);

    int N=1<<n;
#pragma omp parallel shared(pow,shift,N)
    {
#pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            int i0 = i & ~pow;


            int i1 = i | pow;

            int iq = (i & pow) >> shift;

            W[i] = U[iq][0] * V[i0] + U[iq][1] * V[i1];
        }
    }

}

int main(int argc , char** argv) {
    if(argc<4){
        cout<<"not enough arguments";
        return 1;
    }
    omp_set_num_threads(atoi(argv[3]));
    int n = atoi(argv[1]); // number of qubits
    int k = atoi(argv[2]); // operated qubit number
    assert(n>=k);
    struct timeval start,stop;

    //printf("%d",omp_get_num_threads());

    complexd *V = generate_condition(n);
    unsigned long long index = 1LLU<<n;
    complexd *W = new complexd[index];
    complexd U[2][2];
    U[0][0] = 1/sqrt(2);
    U[0][1] = 1/sqrt(2);
    U[1][0] = 1/sqrt(2);
    U[1][1] =  - 1/sqrt(2);
    gettimeofday(&start,NULL);
    OneQubitEvolution(V,W,U,n,k);

    gettimeofday(&stop,NULL);

    printf("LIFETIME:%lf s\n",(float)((stop.tv_sec - start.tv_sec)*1000000 + stop.tv_usec - start.tv_usec)/1000000);
    delete [] V;
    delete [] W;

}
