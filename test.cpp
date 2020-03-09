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

void OneQubitEvolution(complexd* V,complexd* W,complexd U[2][2],int n, int k){
    int shift = n-k;
    int pow=1<<(shift);

    int N=1<<n;
#pragma omp parallel firstprivate(pow,shift,N)
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

unsigned  int compare(complexd* W1,complexd* W2,int n){
    //n - number of cubits
    unsigned  int size = 1U<<n;
    for (unsigned int i = 0; i <size ; ++i) {
        if (W1[i]!=W2[i]){
            return i;
        }
    }
    return 0;
}


int main(int argc , char** argv) {
    if(argc<3){
        cout<<"not enough arguments";
        return 1;
    }
    int n = atoi(argv[1]); // number of qubits
    int k = atoi(argv[2]); // operated qubit number
    assert(n>=k);
    FILE *file = fopen(argv[3],"rb");
    struct timeval start,stop;
    long long unsigned qsize = 1LLU<<n; // pow(2,n) for a condition-vector
    auto *V = new complexd[qsize];
    auto *W1 = new complexd[qsize];

    for (unsigned int i = 0; i <qsize ; ++i) {
        double _real,_im;
        fread(&_real,sizeof(double),1,file);
        fread(&_im,sizeof(double),1,file);
        V[i].real(_real);
        V[i].imag(_im);

    }
    for (unsigned int i = 0; i <qsize ; ++i) {
        double _real,_im;
        fread(&_real,sizeof(double),1,file);
        fread(&_im,sizeof(double),1,file);
        W1[i].real(_real);
        W1[i].imag(_im);
    }

    auto *W2 = new complexd[qsize];
    complexd U[2][2];
    U[0][0] = 1/sqrt(2);
    U[0][1] = 1/sqrt(2);
    U[1][0] = 1/sqrt(2);
    U[1][1] =  - 1/sqrt(2);
    OneQubitEvolution(V,W2,U,n,k);
    unsigned int a = compare(W1,W2,n);
    if(a == 0){
        printf("tests passed\n");
    }else{
        printf("error in %d position\n",a);
    }
    delete [] V;
    delete [] W1;
    delete []W2;
    fclose(file);

}