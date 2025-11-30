# 1. Collect Headers
file(GLOB_RECURSE HEADER_LIST CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../include/tt/*.h")

# 2. Define the Library Target
add_library(tinytensor STATIC
        ${HEADER_LIST}

        ../include/tt/jit/Ops.h
        ../include/tt/jit/Graph.h
        ../include/tt/jit/OpNode.h

        tensor/backend/jit/storage_jit.h
        tensor/backend/jit/backend_jit.h
        tensor/backend/jit/backend_jit.cpp

        ../include/tt/jit/Visitor.h
        ../include/tt/jit/Compiler.h
        jit/Compiler.cpp
)

add_library(tinytensor::tinytensor ALIAS tinytensor)

# Enforce C++20 on Target
target_compile_features(tinytensor PUBLIC cxx_std_20)

# GENERATE EXPORT HEADER !! dont forget
include(GenerateExportHeader)
generate_export_header(tinytensor
        EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/tt/export.h"
)

# Include Directories
target_include_directories(tinytensor PUBLIC
        # Source includes (your code)
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>"
        "$<INSTALL_INTERFACE:include>"

        # Generated includes (where export.h lives)
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
)

target_include_directories(tinytensor PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(tinytensor SYSTEM PRIVATE "${PROJECT_SOURCE_DIR}/external/libnop/include")

target_compile_definitions(tinytensor PUBLIC
        "$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:TINYTENSOR_STATIC_DEFINE>"
)

if(TT_BUILD_CUDA)
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        target_compile_features(tinytensor PUBLIC cuda_std_20)
        target_compile_definitions(tinytensor PUBLIC TT_CUDA)
        target_link_libraries(tinytensor PRIVATE CUDA::cudart)

        set_target_properties(tinytensor PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_ARCHITECTURES native
        )
    endif()
endif()

add_subdirectory(tensor)
add_subdirectory(common)
add_subdirectory(autograd)
add_subdirectory(nn)
add_subdirectory(optim)
add_subdirectory(data)

