cmake_minimum_required(VERSION 3.5.0)
project(CAB401 VERSION 0.1.0 LANGUAGES C CXX)

# SOURCES
file(GLOB SOURCES "src/*.cpp")

# INCLUDE
include_directories(include)

# LINK
link_directories(lib)

# LIBRARIES
link_libraries(OpenCL)

# EXECUTABLE
add_executable(CAB401 ${SOURCES})
