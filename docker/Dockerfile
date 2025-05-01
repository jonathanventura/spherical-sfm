FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ARG DEBIAN_FRONTEND=noninteractive

ENV TZ="America/Los_Angeles"
RUN apt-get -y update && apt-get -y install \
    wget \
    gdb \
    unzip \
    git \
    cmake \
    ninja-build \
    python3-pip \
    build-essential \
    libboost-all-dev \
    libeigen3-dev \
    libgoogle-glog-dev \
    libgsl-dev \
    libtbb-dev \
    libsuitesparse-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    libcgal-dev -y

WORKDIR /

RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.10.0.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.10.0.zip && \
    unzip opencv.zip && \
    unzip opencv_contrib.zip && \
    mkdir -p build && cd build && \
    cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.10.0/modules \
    -DBUILD_opencv_hdf=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_LIST=calib3d,highgui,features2d,imgproc,flann,videoio \
    ../opencv-4.10.0 && \
    make -j8 && \
    make install
    
RUN wget http://ceres-solver.org/ceres-solver-2.2.0.tar.gz && \
    tar -zxf ceres-solver-2.2.0.tar.gz && \
    mkdir ceres-bin && \
    cd ceres-bin && \
    cmake ../ceres-solver-2.2.0 -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF && \
    make -j3 && \
    make install

RUN git clone https://github.com/vlarsson/PoseLib.git && \
    mkdir PoseLib-build && cd PoseLib-build && \
    cmake ../PoseLib && \
    make && \
    make install

RUN git clone https://github.com/laurentkneip/opengv.git && \
    cd opengv && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j8 && \
    make install

RUN git clone https://github.com/jonathanventura/polynomial.git && \
    mkdir polynomial-build && cd polynomial-build && \
    cmake ../polynomial && \
    make && \
    make install
