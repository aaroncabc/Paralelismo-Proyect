git clone https://github.com/aaroncabc/Paralelismo-Proyect.git
cd Paralelismo-Proyect
docker build -t mpi-python .
docker run --rm mpi-python mpirun -np 4 python3 matmul_mpi.py 500
docker run --rm mpi-python mpirun -np 4 python3 primes_mpi.py 4
