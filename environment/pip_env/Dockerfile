# FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest
FROM mcr.microsoft.com/azureml/minimal-ubuntu22.04-py39-cpu-inference:latest

WORKDIR /

# Should create env and install virtual env, take care of proxies and ssl certs
ENV CONDA_ENV_DIR=/opt/miniconda/envs
COPY requirements.txt .
RUN conda create -n userenv python=3.11
ENV CONDA_ENV_PATH="$CONDA_ENV_DIR/userenv"
ENV PATH="$CONDA_ENV_PATH/bin:$PATH"
RUN pip install -r requirements.txt

CMD ["bash"]