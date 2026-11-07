cmake_minimum_required(VERSION 3.12)
project(scope)

set(CMAKE_CXX_STANDARD 17)
IF(NOT APPLE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
ENDIF()

include_directories(
        ${PROJECT_SOURCE_DIR}/config
        ${PROJECT_SOURCE_DIR}/utility
        ${PROJECT_SOURCE_DIR}/utility/automorphism
        ${PROJECT_SOURCE_DIR}/graph
        ${PROJECT_SOURCE_DIR}/plan
        ${PROJECT_SOURCE_DIR}/counting
)

link_directories(
        ${PROJECT_SOURCE_DIR}/config
        ${PROJECT_SOURCE_DIR}/utility
        ${PROJECT_SOURCE_DIR}/utility/automorphism
        ${PROJECT_SOURCE_DIR}/graph
        ${PROJECT_SOURCE_DIR}/plan
        ${PROJECT_SOURCE_DIR}/counting
)

# link to GLPK. edit the paths based on your environment
IF(NOT APPLE)
    include_directories($ENV{HOME}/.local/include)
    link_directories($ENV{HOME}/.local/lib)
ELSE()
    include_directories(/opt/homebrew/Cellar/glpk/5.0/include)
    link_directories(/opt/homebrew/Cellar/glpk/5.0/lib)
ENDIF()

add_subdirectory(config)
add_subdirectory(graph)
add_subdirectory(utility)
add_subdirectory(executable)
add_subdirectory(plan)
add_subdirectory(counting)

