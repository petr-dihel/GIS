cmake_minimum_required(VERSION 3.0.0)
project(gis VERSION 0.1.0)

#include(CTest)
#enable_testing()

find_package(PROJ CONFIG REQUIRED)
include_directories(${PROJ4_INCLUDE_DIRS})

find_package( OpenCV REQUIRED )

add_executable(gis main.cpp)

target_link_libraries(gis ${OpenCV_LIBS} )
target_link_libraries(gis PROJ::proj)



#set(CPACK_PROJECT_NAME ${PROJECT_NAME})
#set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
#include(CPack)
