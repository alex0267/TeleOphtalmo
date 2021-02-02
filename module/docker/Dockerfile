FROM nvidia/cuda:10.2-base

RUN apt update

# Install python 3.7
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.7 python3.7-dev python3-pip

WORKDIR app/

# Install python project dependencies
COPY requirements.txt .
RUN python3.7 -m pip install --upgrade pip setuptools wheel
RUN python3.7 -m pip install -r requirements.txt

