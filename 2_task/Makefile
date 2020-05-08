all:
	 mpic++ mpi_prac.cpp -fopenmp
submit:
	for n in 20 24 28 30; do \
    		for k in 1 3 $$n; do \
    			for i in 1 2 4 8; do \
    				bsub -W 15 -q normal -o out_$$k-$$n -e err_$$k-$$n OMP_NUM_THREADS=8  mpiexec -n $$i ./a.out -n $$n -k $$k -file_write out.out \
    			done \
    		done \
    	done