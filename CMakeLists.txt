cmake_minimum_required(VERSION 3.14)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
project(cpp_selfplay LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GAME_IMPL "gomoku" CACHE STRING "Game implementation to compile (gomoku or quoridor)")
set_property(CACHE GAME_IMPL PROPERTY STRINGS gomoku quoridor)

if(GAME_IMPL STREQUAL "gomoku")
  set(GAME_SOURCE games/gomoku.cpp)
  add_compile_definitions(GAME_IMPL_GOMOKU)
elseif(GAME_IMPL STREQUAL "quoridor")
  set(GAME_SOURCE games/quoridor.cpp)
  add_compile_definitions(GAME_IMPL_QUORIDOR)
else()
  message(FATAL_ERROR "Unsupported GAME_IMPL='${GAME_IMPL}'. Use gomoku or quoridor.")
endif()

add_executable(
  cpp_selfplay
  main.cpp
  search.cpp
  onnx_infer.cpp
  ${GAME_SOURCE}
)

target_compile_options(cpp_selfplay PRIVATE -O3 -march=native -ffast-math -flto)

if(NOT DEFINED ONNXRUNTIME_DIR)
  message(FATAL_ERROR "Set ONNXRUNTIME_DIR to your onnxruntime installation root")
endif()

target_include_directories(cpp_selfplay PRIVATE ${ONNXRUNTIME_DIR}/include)

find_library(
  ONNXRUNTIME_LIB
  NAMES onnxruntime
  PATHS ${ONNXRUNTIME_DIR}/lib
  NO_DEFAULT_PATH
  REQUIRED
)

find_package(Threads REQUIRED)
target_link_libraries(cpp_selfplay PRIVATE ${ONNXRUNTIME_LIB} Threads::Threads)
