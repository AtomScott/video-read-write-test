OPENCV_DIR=/home/atom/opencv

cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=/home/atom/opencv_contrib/modules \
    -D WITH_CUDA=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D WITH_NVCUVID=ON \
    -D WITH_NVCUVENC=ON \
    -D NVCUVID_INCLUDE_DIR=/home/atom/NVIDIA_Video_Codec_SDK/Interface \
    -D NVCUVENC_INCLUDE_DIR=/home/atom/NVIDIA_Video_Codec_SDK/Interface \
    -D NVCUVID_LIBRARY=/home/atom/NVIDIA_Video_Codec_SDK/Lib/linux/x86_64/libnvcuvid.so \
    -D NVCUVENC_LIBRARY=/home/atom/NVIDIA_Video_Codec_SDK/Lib/linux/x86_64/libnvencodeapi.so \
    -D BUILD_opencv_python3=ON \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_TESTS=OFF \
    -D CUDA_ARCH_BIN=8.6 \
    -D CUDA_ARCH_PTX=8.6 \
    -D CMAKE_CXX_STANDARD=11 \
    -D ENABLE_PRECOMPILED_HEADERS=OFF \
    -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -D WITH_CUFFT=OFF \
    -D WITH_CUBLAS=ON \
    -D CUDA_NVCC_FLAGS="-D_FORCE_INLINES" \
    -D PYTHON3_EXECUTABLE=/home/atom/video-read-write-test/.venv/bin/python \
    -D PYTHON3_INCLUDE_DIR=/home/atom/video-read-write-test/.venv/include/python3.x \
    -D PYTHON3_PACKAGES_PATH=/home/atom/video-read-write-test/.venv/lib/python3.x/site-packages \
    $OPENCV_DIR

make -j$(nproc)
make install

