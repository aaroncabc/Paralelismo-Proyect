FROM ubuntu:22.04

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    build-essential \
    openmpi-bin openmpi-common libopenmpi-dev \
    && pip3 install mpi4py numpy

# Crear usuario sin privilegios
RUN useradd -ms /bin/bash mpiuser
USER mpiuser

# Crear carpeta de trabajo y copiar archivos
WORKDIR /home/mpiuser/app
COPY --chown=mpiuser:mpiuser . .

# Comando por defecto
CMD ["/bin/bash"]
