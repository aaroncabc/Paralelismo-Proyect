from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Tamaño fijo de la matriz
N = 100

# Calcular distribución de filas por proceso
rows_per_proc = [N // size + (1 if i < N % size else 0) for i in range(size)]
displs_rows = [sum(rows_per_proc[:i]) for i in range(size)]

# Buffers locales
local_rows = rows_per_proc[rank]
local_A = np.empty((local_rows, N), dtype=np.float64)
B = np.empty((N, N), dtype=np.float64)  # buffer común a todos los procesos

if rank == 0:
    A = np.random.rand(N, N)
    B[:, :] = np.random.rand(N, N)

    # Preparar flat_A y buffers para Scatterv
    flat_A = A.ravel()
    sendcounts = [rows * N for rows in rows_per_proc]
    displs = [sum(sendcounts[:i]) for i in range(size)]
else:
    flat_A = None
    sendcounts = [0] * size
    displs = [0] * size

# Asegurar sincronía y broadcast de metadatos
sendcounts = comm.bcast(sendcounts, root=0)
displs = comm.bcast(displs, root=0)

# Broadcast de B
comm.Bcast(B, root=0)
print(f"[Rank {rank}] recibió B con shape {B.shape}")

# Validación opcional
assert local_A.size == sendcounts[rank], f"Rank {rank} local_A.size != sendcounts[rank]"

# Scatterv de A
comm.Scatterv([flat_A, sendcounts, displs, MPI.DOUBLE], local_A.ravel(), root=0)
print(f"[Rank {rank}] recibió local_A con shape: {local_A.shape}")

# Multiplicación local
comm.Barrier()
start = MPI.Wtime()
local_C = np.matmul(local_A, B)
end = MPI.Wtime()

# Preparar Gatherv
recvcounts = [rows * N for rows in rows_per_proc] if rank == 0 else [0] * size
recvdispls = [sum(recvcounts[:i]) for i in range(size)] if rank == 0 else [0] * size

# Broadcast para que todos conozcan las dimensiones (por seguridad)
recvcounts = comm.bcast(recvcounts, root=0)
recvdispls = comm.bcast(recvdispls, root=0)

# Preparar buffer destino en rank 0
if rank == 0:
    flat_C = np.empty(N * N, dtype=np.float64)
else:
    flat_C = None

# Sincronización antes de recolección
comm.Barrier()
print(f"[Rank {rank}] antes de Gatherv")

comm.Gatherv(local_C.ravel(), [flat_C, recvcounts, recvdispls, MPI.DOUBLE], root=0)

print(f"[Rank {rank}] después de Gatherv")

# Rank 0 reconstruye C
if rank == 0:
    C = flat_C.reshape((N, N))
    print(f"[Rank {rank}] matriz final C con shape: {C.shape}")
    print(f"[Rank {rank}] Tiempo total MPI con {size} procesos: {end - start:.6f} segundos")
