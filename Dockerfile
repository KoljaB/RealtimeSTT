FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 as gpu

WORKDIR /app

RUN apt-get update -y && \
  apt-get install -y python3 python3-pip portaudio19-dev

RUN pip3 install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128

COPY requirements-gpu* /app/
RUN pip3 install -r /app/requirements-gpu-torch.txt && \
  pip3 install -r /app/requirements-gpu.txt

RUN mkdir example_browserclient
COPY example_browserclient/server.py /app/example_browserclient/server.py
COPY RealtimeSTT /app/RealtimeSTT

EXPOSE 9001
ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN export PYTHONPATH="${PYTHONPATH}:/app"
CMD ["python3", "example_browserclient/server.py"]

# --------------------------------------------

FROM ubuntu:24.04 as cpu

WORKDIR /app

RUN apt-get update -y && \
  apt-get install -y python3 python3-pip portaudio19-dev

RUN pip3 install torch==2.7.1 torchaudio==2.7.1

COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

EXPOSE 9001
ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN export PYTHONPATH="${PYTHONPATH}:/app"
CMD ["python3", "example_browserclient/server.py"]
