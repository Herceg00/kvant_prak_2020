#include "gates_library.h"
#include "conditions.h"

int main(int argc, char** argv){
    bool file_read = false;
    bool test_flag = false;
    char *input, *output, *test_file;
    unsigned k, n, l;
    for (int i = 1; i < argc; i++) {
        string option(argv[i]);
        if (option.compare("n") == 0) {
            n = atoi(argv[++i]);
        }

        if ((option.compare("k") == 0)) {
            k = atoi(argv[++i]);
        }

        if ((option.compare("l") == 0)) {
            l = atoi(argv[++i]);
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
    double thetta = 0.5;
//    double begin = MPI_Wtime();
    CROT(k,l,V,rank,size,n,thetta);
//    double end = MPI_Wtime();
//    std::cout << "The process took " << end - begin << " seconds to run." << std::endl;
    MPI_Finalize();
}