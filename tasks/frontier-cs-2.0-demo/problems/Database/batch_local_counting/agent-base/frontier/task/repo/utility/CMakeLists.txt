set(UTILITY_SRC
        command_parser.cpp
        compute_set_intersection.cpp utils.cpp clique.cpp)

add_library(utility SHARED
        ${UTILITY_SRC})