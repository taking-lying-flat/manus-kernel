cmake_minimum_required(VERSION 3.20)

project(mnist-kernel LANGUAGES C CXX)

if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE
   AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MNIST_NATIVE_ARCH "Optimize for the current CPU" ON)
option(MNIST_BUILD_LEGACY_C "Build the legacy C implementation" OFF)

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(mnist-kernel mnist.cpp)
target_compile_features(mnist-kernel PRIVATE cxx_std_20)
target_include_directories(
    mnist-kernel PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(mnist-kernel PRIVATE OpenMP::OpenMP_CXX)
target_compile_definitions(
    mnist-kernel PRIVATE
    MNIST_DEFAULT_DATA_DIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mnist-kernel PRIVATE -Wall -Wextra -Wpedantic)
    if(MNIST_NATIVE_ARCH)
        target_compile_options(mnist-kernel PRIVATE -march=native)
    endif()
endif()

if(MNIST_BUILD_LEGACY_C)
    add_executable(mnist_legacy mnist.c)
    set_target_properties(mnist_legacy PROPERTIES OUTPUT_NAME mnist-c)
    if(NOT MSVC)
        target_link_libraries(mnist_legacy PRIVATE m)
    endif()
endif()
