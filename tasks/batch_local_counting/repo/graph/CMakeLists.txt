include_directories(../utility/automorphism)
link_directories(../utility/automorphism)
IF(NOT APPLE)
    include_directories($ENV{HOME}/.local/include)
    link_directories($ENV{HOME}/.local/lib)
ELSE()
    include_directories(/opt/homebrew/Cellar/glpk/5.0/include)
    link_directories(/opt/homebrew/Cellar/glpk/5.0/lib)
ENDIF()
set(GRAPH_SRC
        graph.cpp
        )

add_library(graph SHARED ${GRAPH_SRC})
#add_executable(directg.out ../utility/automorphism/directg.c)

target_link_libraries(graph nauty utility glpk)