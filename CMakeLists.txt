cmake_minimum_required(VERSION 3.5)
project(Mask)
set(CMAKE_CXX_STANDARD 11)
set(Cudart_LIBS "/usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudart.so")
enable_language(CUDA)
find_package(CUDA REQUIRED)
find_package(OpenCV 4 REQUIRED)
include_directories(
		Yolo
        ${OpenCV_INCLUDE_DIRS}
		/usr/local/cuda/include
)
add_library(
		Yolo
		Yolo/yolov5.cpp
)

cuda_add_library(myplugins SHARED ./Yolo/yololayer.cu)


add_executable(Mask
		       Mask_Detection.cpp
		       Yolo/yolov5.cpp
        )
target_link_libraries(
		Mask
		nvinfer
		cudart
		myplugins
		/usr/local/cuda/lib64
		${Cudart_LIBS}
		${OpenCV_LIBS}
        )
