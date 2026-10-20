include_directories(../utility/automorphism)
link_directories(../utility/automorphism)
set (PREPROCESS_SRC
        prepare.cpp)
set (SCOPE_SRC
        scope.cpp)
add_executable(preprocess.out ${PREPROCESS_SRC})
add_executable(scope.out ${SCOPE_SRC})
target_link_libraries(preprocess.out graph counting utility)
target_link_libraries(scope.out graph plan counting utility)
