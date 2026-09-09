set(TARGET llama-rest)
add_executable(${TARGET}
    rest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/rest-store.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/libsais.c
)
install(TARGETS ${TARGET} RUNTIME)
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../common)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../../common/libsais.c PROPERTIES LANGUAGE C)
