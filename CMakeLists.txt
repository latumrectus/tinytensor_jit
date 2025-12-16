cmake_minimum_required(VERSION 3.25)

project(tinytensor VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 3. Build Options
option(TT_BUILD_CUDA "Build tinytensor with cuda backend support" OFF)
option(TT_BUILD_TESTS "Build tinytensor tests" OFF)
option(TT_BUILD_EXAMPLES "Build tinytensor examples" OFF)

# 4. Find LLVM / MLIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# 5. LLVM / MLIR CMake utilities
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# 6. Compiler flags
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# 7. Subdirectories
add_subdirectory(tinytensor)

if (TT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if (TT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()