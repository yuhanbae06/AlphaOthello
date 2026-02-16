cmake_minimum_required(VERSION 3.10)

project(Test)

set(CMAKE_PREFIX_PATH "C:/libtorch") 

find_package(Torch REQUIRED)

if(MSVC)
  add_compile_options(/wd4244 /wd4267)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(main main.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")

set_property(TARGET main PROPERTY CXX_STANDARD 17)