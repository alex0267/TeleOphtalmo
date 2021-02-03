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

# Install MaskRCNN
RUN apt-get install -y git
RUN git clone https://www.github.com/matterport/Mask_RCNN.git
RUN cd Mask_RCNN && python3.7 setup.py -q install

# Install OpenCV deps
RUN apt-get install ffmpeg libsm6 libxext6 -y

# Install documentation dependencies
RUN python3.7 -m pip install sphinx sphinx-rtd-theme
RUN mkdir docs
COPY docs/ ./docs

# Download MRCNN transfer learning weights
RUN apt-get install wget -y
RUN wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

# Copy module into container
COPY *.py ./

# Build documentation
RUN cd docs && make html
EXPOSE 8000

ENTRYPOINT ["python3.7", "main.py"]
CMD docs
