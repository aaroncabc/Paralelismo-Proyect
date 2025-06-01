FROM ubuntu:22.04

# Instala dependencias
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    build-essential \
    openmpi-bin openmpi-common libopenmpi-dev \
    && pip3 install mpi4py numpy

# Copia los scripts al contenedor
WORKDIR /app
COPY . /app

# Comando por defecto
CMD ["/bin/bash"]
