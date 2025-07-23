# https://www.learnopencv.com/install-opencv3-on-ubuntu
# https://docs.opencv.org/3.4/d6/d15/tutorial_building_tegra_cuda.html

# FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04
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
    libcgal-dev \
    python3-pandas

WORKDIR /opencv

# Install OpenCV
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

RUN wget https://github.com/colmap/colmap/archive/refs/tags/3.11.1.zip && \
    unzip 3.11.1.zip && \
    cd colmap-3.11.1 && \
    mkdir build && \
    cd build && \
    cmake .. -DGUI_ENABLED=OFF -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=70 && \
    make -j8 && \
    make install

# RUN git clone https://github.com/kthohr/optim.git && \
#     cd optim && \
#     git submodule update --init && \
#     export EIGEN_INCLUDE_PATH=/usr/include/eigen3 && \
#     ./configure -i "/usr/local" -l eigen -p && \
#     make && \
#     make install

# RUN git clone https://github.com/AIBluefisher/GraphOptim.git && \
RUN git clone https://github.com/jonathanventura/GraphOptim.git && \
    cd GraphOptim  && \
    mkdir build && cd build && \
    cmake .. && \
    make -j8 && \
    make install
