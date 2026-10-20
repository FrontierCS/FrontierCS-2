include_directories(../utility/automorphism)
link_directories(../utility/automorphism)

set(PLAN_SRC
        tree.cpp
        cost_estimator.cpp
        decompose.cpp)

add_library(plan SHARED ${PLAN_SRC})
target_link_libraries(plan graph utility)
