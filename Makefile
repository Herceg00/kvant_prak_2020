all:
	g++ main.cpp -fopenmp -o prog
submit:
	for n in 20 24 28 30; do \
    		for k in 1 3 $$n; do \
    			for i in 1 2 4 8; do \
    				bsub -W 15 -q normal -o out_$$k-$$n -e err_$$k-$$n OMP_NUM_THREADS=$$i ./prog $$n $$k; \
    			done \
    		done \
    	done
testing:
	g++ test.cpp -fopenmp -o test
	./test 8 5 test8_5
	./test 16 3 test16_3
