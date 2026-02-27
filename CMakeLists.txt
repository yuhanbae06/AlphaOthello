cmake_minimum_required(VERSION 3.14)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
project(cpp_selfplay LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GAME_IMPL "gomoku" CACHE STRING "Game implementation to compile (gomoku, quoridor5, quoridor7, quoridor9)")
set_property(CACHE GAME_IMPL PROPERTY STRINGS gomoku quoridor quoridor5 quoridor7 quoridor9)

if(GAME_IMPL STREQUAL "gomoku")
  set(GAME_SOURCE games/gomoku.cpp)
  add_compile_definitions(GAME_IMPL_GOMOKU)
elseif(GAME_IMPL STREQUAL "quoridor5")
  set(GAME_SOURCE games/quoridor5.cpp)
  add_compile_definitions(GAME_IMPL_QUORIDOR5)
elseif(GAME_IMPL STREQUAL "quoridor" OR GAME_IMPL STREQUAL "quoridor7")
  set(GAME_SOURCE games/quoridor7.cpp)
  add_compile_definitions(GAME_IMPL_QUORIDOR7)
elseif(GAME_IMPL STREQUAL "quoridor9")
  set(GAME_SOURCE games/quoridor9.cpp)
  add_compile_definitions(GAME_IMPL_QUORIDOR9)
else()
  message(FATAL_ERROR "Unsupported GAME_IMPL='${GAME_IMPL}'. Use gomoku, quoridor5, quoridor7, or quoridor9.")
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
