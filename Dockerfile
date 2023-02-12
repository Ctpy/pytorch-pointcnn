FROM continuumio/anaconda3:latest

WORKDIR /app

# Create the environment:
COPY . .
# COPY ../pointcnn/ .
RUN conda init
RUN conda env create -n pointcnn
RUN conda activate -n pointcnn
RUN conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
RUN conda install pytorch-lightning -c conda-forge
RUN conda install pyg -c pyg
# Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "pointcnn", "/bin/bash", "-c"]

# The code to run when container is started:
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "pointcnn", "python", "main.py"]
