if (GGML_TILELANG)
    option(GGML_TILELANG_USE_AOT_KERNELS
           "ggml: use TileLang generated AOT kernels"
           OFF)
    option(GGML_TILELANG_EXPERIMENTAL_STREAM_INTEROP
           "ggml: enable experimental CUDA stream interop for TileLang"
           ON)

    if (GGML_CUDA)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        ggml_add_backend_library(ggml-tilelang
            ggml-tilelang.cpp
            tilelang-cuda-util.cu
        )

        set(GGML_TILELANG_AOT_SOURCES
            generated/tilelang_f16_gemv.cu
            generated/tilelang_f16_gemv_launcher.cu
            generated/tilelang_q8_0_gemv.cu
            generated/tilelang_q8_0_gemv_launcher.cu
        )

        if (GGML_TILELANG_USE_AOT_KERNELS)
            foreach(GGML_TILELANG_AOT_SOURCE IN LISTS GGML_TILELANG_AOT_SOURCES)
                if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${GGML_TILELANG_AOT_SOURCE}")
                    message(FATAL_ERROR
                        "GGML_TILELANG_USE_AOT_KERNELS=ON but ${GGML_TILELANG_AOT_SOURCE} is missing. "
                        "Run ggml/src/ggml-tilelang/kernels/gen_kernels.py first.")
                endif()
            endforeach()
            target_sources(ggml-tilelang PRIVATE ${GGML_TILELANG_AOT_SOURCES})
            target_compile_definitions(ggml-tilelang PRIVATE GGML_TILELANG_USE_AOT_KERNELS)
        else()
            target_sources(ggml-tilelang PRIVATE
                tilelang-kernels-cuda-stub.cu
                tilelang-q8-0-gemv-cuda-stub.cu
            )
        endif()

        get_target_property(CUDA_ARCHS ggml-cuda CUDA_ARCHITECTURES)
        if (CUDA_ARCHS)
            set_target_properties(ggml-tilelang PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCHS}")
        endif()
        target_link_libraries(ggml-tilelang PRIVATE ggml-cuda)
        target_compile_definitions(ggml-tilelang PRIVATE GGML_TILELANG_CUDA)
        if (GGML_TILELANG_EXPERIMENTAL_STREAM_INTEROP)
            target_compile_definitions(ggml-tilelang PRIVATE GGML_TILELANG_EXPERIMENTAL_STREAM_INTEROP)
        endif()

        if (GGML_TILELANG_USE_AOT_KERNELS)
            add_executable(tilelang-q8-0-microbench
                tilelang-q8-0-microbench.cu
            )
            target_link_libraries(tilelang-q8-0-microbench PRIVATE ggml-tilelang CUDA::cudart_static)
            target_include_directories(tilelang-q8-0-microbench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
            if (CUDA_ARCHS)
                set_target_properties(tilelang-q8-0-microbench PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCHS}")
            endif()
        endif()
    else()
        ggml_add_backend_library(ggml-tilelang
            ggml-tilelang.cpp
        )
    endif()

    if (GGML_HIP)
        target_link_libraries(ggml-tilelang PRIVATE ggml-hip)
    endif()
endif()
