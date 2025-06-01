from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 100

# Distribuir filas
rows_per_proc = [N // size + (1 if i < N % size else 0) for i in range(size)]
displs_rows = [sum(rows_per_proc[:i]) for i in range(size)]
local_rows = rows_per_proc[rank]

# Buffers locales
local_A = np.empty((local_rows, N), dtype=np.float64)
B = np.empty((N, N), dtype=np.float64)

if rank == 0:
    A = np.random.rand(N, N)
    B[:, :] = np.random.rand(N, N)
    flat_A = A.flatten()
    sendcounts = [rows * N for rows in rows_per_proc]
    displs = [sum(sendcounts[:i]) for i in range(size)]
else:
    flat_A = None
    sendcounts = None
    displs = None

# Broadcast metadatos
sendcounts = comm.bcast(sendcounts, root=0)
displs = comm.bcast(displs, root=0)

# Broadcast de B
comm.Bcast(B, root=0)

# Scatterv de A
comm.Scatterv([flat_A, sendcounts, displs, MPI.DOUBLE], local_A.ravel(), root=0)

# Multiplicación local
comm.Barrier()
start = MPI.Wtime()
local_C = np.matmul(local_A, B)
end = MPI.Wtime()

# Recolección manual con Ssend / Recv
if rank == 0:
    C = np.empty((N, N), dtype=np.float64)
    # Copiar los resultados locales propios
    C[0:local_rows, :] = local_C

    for i in range(1, size):
        rows_i = rows_per_proc[i]
        buffer = np.empty((rows_i, N), dtype=np.float64)
        comm.Recv(buffer, source=i, tag=77)
        C[displs_rows[i]:displs_rows[i]+rows_i, :] = buffer

    print(f"[Rank 0] Matriz final C con shape {C.shape}")
    print(f"[Rank 0] Tiempo total MPI con {size} procesos: {end - start:.6f} segundos")

else:
    # Enviar resultado local al root
    comm.Ssend(local_C, dest=0, tag=77)
