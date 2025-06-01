from mpi4py import MPI
import numpy as np
import sys
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = int(sys.argv[1]) if rank == 0 else None
N = comm.bcast(N, root=0)

# Cada proceso genera sus datos
A = np.empty((N, N), dtype=np.float64) if rank == 0 else None
B = np.empty((N, N), dtype=np.float64)

if rank == 0:
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)

# Broadcast B completa
comm.Bcast(B, root=0)

# Scatter rows of A
rows_per_proc = N // size
local_A = np.empty((rows_per_proc, N), dtype=np.float64)
comm.Scatter(A, local_A, root=0)

start = MPI.Wtime()
local_C = np.matmul(local_A, B)
end = MPI.Wtime()

# Gather results
C = None
if rank == 0:
    C = np.empty((N, N), dtype=np.float64)
comm.Gather(local_C, C, root=0)

if rank == 0:
    print(f"Tiempo total MPI (rango {size}): {end - start:.6f} segundos")
