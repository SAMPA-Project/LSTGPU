cmake_minimum_required(VERSION 3.18)
project(LSTGPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES)
  # Turing/Ampere/Ada by default -- override for your GPU, e.g.:
  #   cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

find_package(TIFF REQUIRED)
find_package(Threads REQUIRED)

add_executable(LSTGPU
  src/main.cpp
  src/surface_raster.cpp
  src/materials.cpp
  src/meteo.cpp
  src/model_params.cpp
  src/geotiff_io.cpp
  src/output_writer.cpp
  src/sky_bins.cpp
  src/cuda/device_globals.cu
  src/cuda/device_upload.cu
  src/cuda/lst_pipeline.cu
  src/cuda/kernels/viewshed_kernels.cu
  src/cuda/kernels/irradiance_kernel.cu
  src/cuda/kernels/reflection_kernels.cu
  src/cuda/kernels/wind_kernel.cu
  src/cuda/kernels/seb_kernel.cu
)

target_include_directories(LSTGPU PRIVATE include)
target_link_libraries(LSTGPU PRIVATE TIFF::TIFF Threads::Threads)

if(MSVC)
  target_compile_definitions(LSTGPU PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()

# The kernels share __constant__/__device__ globals (src/cuda/device_globals.cu) across several .cu files
set_target_properties(LSTGPU PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
