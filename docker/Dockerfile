# https://www.learnopencv.com/install-opencv3-on-ubuntu
# https://docs.opencv.org/3.4/d6/d15/tutorial_building_tegra_cuda.html

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

ENV TZ="America/Los_Angeles"
RUN apt-get -y update && apt-get -y install git wget gdb libboost-all-dev cmake build-essential libeigen3-dev libgoogle-glog-dev libgsl-dev colmap libsuitesparse-dev libtbb-dev unzip

WORKDIR /opencv

# Install OpenCV
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.10.0.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.10.0.zip && \
    unzip opencv.zip && \
    unzip opencv_contrib.zip && \
    mkdir -p build && cd build && \
    cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.10.0/modules ../opencv-4.10.0 && \
    make -j8 && \
    make install

WORKDIR /

# Install ceres-solver
RUN wget http://ceres-solver.org/ceres-solver-1.14.0.tar.gz && \
    tar -zxf ceres-solver-1.14.0.tar.gz && \
    mkdir ceres-bin && \
    cd ceres-bin && \
    cmake ../ceres-solver-1.14.0 -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF && \
    make -j3 && \
    make install

RUN git clone https://github.com/AIBluefisher/GraphOptim.git && \
    cd GraphOptim  && \
    mkdir build && cd build && \
    cmake .. && \
    make -j8 && \
    make install


RUN git clone https://github.com/jonathanventura/polynomial.git && \
    mkdir polynomial-build && cd polynomial-build && \
    cmake ../polynomial && \
    make && \
    make install

RUN git clone https://github.com/vlarsson/PoseLib.git && \
    mkdir PoseLib-build && cd PoseLib-build && \
    cmake ../PoseLib && \
    make && \
    make install



RUN apt-get install \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    libcgal-dev -y

#RUN git clone https://github.com/colmap/colmap.git &&\ 
    #cd colmap && \
    #mkdir build && \
    #cd build && \
    #cmake .. -DGUI_ENABLED=OFF -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=70 && \
    #make -j8 && \
    #make install

RUN wget https://github.com/colmap/colmap/archive/refs/tags/3.9.1.zip && \
    unzip 3.9.1.zip && \
    cd colmap-3.9.1 && \
    mkdir build && \
    cd build && \
    cmake .. -DGUI_ENABLED=OFF -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=70 && \
    make -j8 && \
    make install

RUN apt-get install python3-pip -y

RUN git clone https://github.com/laurentkneip/opengv.git && \
    cd opengv && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j8 && \
    make install

RUN pip install numpy torch
RUN pip install pandas
