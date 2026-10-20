include_directories(../utility/automorphism)
link_directories(../utility/automorphism)

set (COUNTING_SRC
        equation.cpp execution.cpp ../executable/command.cpp forest.cpp forest.h triangle.cpp aggregation.cpp aggregation.h)

add_library(counting ${COUNTING_SRC})
target_link_libraries(counting graph plan utility)
