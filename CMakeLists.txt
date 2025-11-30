cmake_minimum_required(VERSION 3.25)

# 1. Define Project and Global Languages
project(tinytensor VERSION 1.0.0 LANGUAGES CXX)

# 2. Enforce C++20 Globally (Fixes CLion LSP & std::format)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 3. Build Options
option(TT_BUILD_CUDA "Build tinytensor with cuda backend support" OFF)
option(TT_BUILD_TESTS "Build tinytensor tests" OFF)
option(TT_BUILD_EXAMPLES "Build tinytensor examples" OFF)

# 4. Compiler Flags (Optimizations)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# 5. Add Main Library Subdirectory
add_subdirectory(tinytensor)

# 6. Optional Components
if (TT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if (TT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()