from mpi4py import MPI
import numpy as np
import sys
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = int(sys.argv[1]) if rank == 0 else None
N = comm.bcast(N, root=0)

rows_per_proc = N // size

local_A = np.empty((rows_per_proc, N), dtype=np.float64)
B = np.empty((N, N), dtype=np.float64)

if rank == 0:
    A = np.random.rand(N, N)
    B[:] = np.random.rand(N, N)
    flat_A = A.reshape(-1)
else:
    flat_A = None

comm.Bcast(B, root=0)
comm.Scatter(flat_A, local_A.reshape(-1), root=0)

start = MPI.Wtime()
local_C = np.matmul(local_A, B)
end = MPI.Wtime()

if rank == 0:
    flat_C = np.empty(N * N, dtype=np.float64)
else:
    flat_C = None

comm.Gather(local_C.reshape(-1), flat_C, root=0)

if rank == 0:
    C = flat_C.reshape((N, N))
    print(f"Tiempo total MPI (rango {size}): {end - start:.6f} segundos")
