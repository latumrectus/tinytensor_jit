# 1. Collect Headers
file(GLOB_RECURSE HEADER_LIST CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/../include/tt/*.h"
)

# 2. Define the Library Target
add_library(tinytensor STATIC
        ${HEADER_LIST}

        ../include/tt/jit/Ops.h
        ../include/tt/jit/Graph.h
        ../include/tt/jit/OpNode.h

        tensor/backend/jit/storage_jit.h
        tensor/backend/jit/JITGraphBuilder.h
        tensor/backend/jit/JITGraphBuilder.cpp

        ../include/tt/jit/Visitor.h
        ../include/tt/jit/Compiler.h
        jit/Compiler.cpp
)

add_library(tinytensor::tinytensor ALIAS tinytensor)

# 3. Enforce C++20
target_compile_features(tinytensor PUBLIC cxx_std_20)

# 4. Export header
include(GenerateExportHeader)
generate_export_header(tinytensor
        EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/tt/export.h"
)

# 5. Include directories
target_include_directories(tinytensor
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
        PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(tinytensor
        SYSTEM PRIVATE
        "${PROJECT_SOURCE_DIR}/external/libnop/include"
)

# 6. MLIR / LLVM
target_link_libraries(tinytensor
        PRIVATE
        MLIRIR
        MLIRSupport
        MLIRFuncDialect
        MLIRTosaDialect
        MLIRParser
        MLIRPass
        MLIRExecutionEngine
        MLIRTargetLLVMIRExport
        LLVMCore
        LLVMSupport
)

target_include_directories(tinytensor
        SYSTEM PRIVATE
        ${MLIR_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS}
)

# 7. Compile definitions
target_compile_definitions(tinytensor PUBLIC
        "$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:TINYTENSOR_STATIC_DEFINE>"
)

# 8. CUDA
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

# 9. Subcomponents
add_subdirectory(tensor)
add_subdirectory(common)
add_subdirectory(autograd)
add_subdirectory(nn)
add_subdirectory(optim)
add_subdirectory(data)

