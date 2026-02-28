cmake_minimum_required(VERSION 3.12)
project(quoridor_viz)

# 1. pybind11 경로 찾기 (pip 설치 시 python 명령어로 경로 추출)
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})

if(GAME_IMPL STREQUAL "quoridor5")
    set(GAME_SOURCE ../games/quoridor5.cpp)
    add_compile_definitions(GAME_IMPL_QUORIDOR5)
    set(PY_MODULE_NAME "quoridor_engine_5")
elseif(GAME_IMPL STREQUAL "quoridor7")
    set(GAME_SOURCE ../games/quoridor7.cpp)
    add_compile_definitions(GAME_IMPL_QUORIDOR7)
    set(PY_MODULE_NAME "quoridor_engine_7")
elseif(GAME_IMPL STREQUAL "quoridor" OR GAME_IMPL STREQUAL "quoridor9")
    set(GAME_SOURCE ../games/quoridor9.cpp)
    add_compile_definitions(GAME_IMPL_QUORIDOR9)
    set(PY_MODULE_NAME "quoridor_engine_9")
else()
    message(FATAL_ERROR "Unsupported GAME_IMPL='${GAME_IMPL}'. Use quoridor5, quoridor7, or quoridor9.")
endif() 

add_compile_definitions(MODULE_NAME=${PY_MODULE_NAME})

# 2. 파이썬 모듈 생성
pybind11_add_module(${PY_MODULE_NAME}
    ../binding.cpp 
    ${GAME_SOURCE}
    ../onnx_infer.cpp
    ../search.cpp
)

# 3. 헤더 및 라이브러리 경로 설정
target_include_directories(${PY_MODULE_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${ONNXRUNTIME_DIR}/include
)

find_library(
  ONNXRUNTIME_LIB
  NAMES onnxruntime
  PATHS ${ONNXRUNTIME_DIR}/lib
  NO_DEFAULT_PATH
  REQUIRED
)

target_link_libraries(${PY_MODULE_NAME} PRIVATE 
    ${ONNXRUNTIME_LIB}
)

# 4. 출력 위치 설정 (.so 파일이 web_play 폴더 바로 아래 생성되도록)
set_target_properties(${PY_MODULE_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)