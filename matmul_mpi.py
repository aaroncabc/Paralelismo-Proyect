from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Tamaño fijo de la matriz
N = 100

# Calcular distribución de filas
rows_per_proc = [N // size + (1 if i < N % size else 0) for i in range(size)]
displs_rows = [sum(rows_per_proc[:i]) for i in range(size)]

# Buffers locales
local_rows = rows_per_proc[rank]
local_A = np.empty((local_rows, N), dtype=np.float64)
B = np.empty((N, N), dtype=np.float64)

if rank == 0:
    A = np.random.rand(N, N)
    B[:, :] = np.random.rand(N, N)
    flat_A = A.flatten()
    sendcounts = [r * N for r in rows_per_proc]
    displs = [sum(sendcounts[:i]) for i in range(size)]
else:
    flat_A = None
    sendcounts = None
    displs = None

sendcounts = comm.bcast(sendcounts, root=0)
displs = comm.bcast(displs, root=0)

# Broadcast de B
comm.Bcast(B, root=0)
print(f"[Rank {rank}] recibió matriz B con shape: {B.shape}")

# Scatterv de A
comm.Scatterv([flat_A, sendcounts, displs, MPI.DOUBLE], local_A.flatten(), root=0)

print(f"[Rank {rank}] recibió local_A con shape: {local_A.shape}")

# Multiplicación local
start = MPI.Wtime()
local_C = np.matmul(local_A, B)
end = MPI.Wtime()

# Preparar recolección
recvcounts = [r * N for r in rows_per_proc]
recvdispls = [sum(recvcounts[:i]) for i in range(size)]

if rank == 0:
    flat_C = np.empty(N * N, dtype=np.float64)
else:
    flat_C = None

comm.Gatherv(local_C.flatten(), [flat_C, recvcounts, recvdispls, MPI.DOUBLE], root=0)

if rank == 0:
    C = flat_C.reshape((N, N))
    print(f"[Rank {rank}] Tiempo total MPI (rango {size}): {end - start:.6f} segundos")
