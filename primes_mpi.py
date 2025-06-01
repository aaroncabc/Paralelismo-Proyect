from mpi4py import MPI
import sys
import time
import math

def is_prime(n):
    if n < 2: return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0: return False
    return True

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

D = int(sys.argv[1]) if rank == 0 else None
D = comm.bcast(D, root=0)

low = 10**(D-1)
high = 10**D

chunk = (high - low) // size
start_val = low + rank * chunk
end_val = high if rank == size - 1 else start_val + chunk

start = MPI.Wtime()
local_primes = [n for n in range(start_val, end_val) if is_prime(n)]
end = MPI.Wtime()

all_counts = comm.gather(len(local_primes), root=0)

if rank == 0:
    print(f"Total de primos: {sum(all_counts)}")
    print(f"Tiempo MPI (rango {size}): {end - start:.6f} segundos")
