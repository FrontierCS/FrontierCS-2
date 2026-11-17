cmake_minimum_required(VERSION 3.16...3.27)

# -- Configure project compiler options
project(spectrum LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS true)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(ENV{CC}     ${CMAKE_C_COMPILER})
set(ENV{CXX}    ${CMAKE_CXX_COMPILER})

option(NDEBUG ON)

find_package(Threads REQUIRED) # we need threads support for OS. 

# -- Manage External Dependencies Using Github

function(Execute)
    cmake_parse_arguments(EX "OUTPUT_QUIET;ERROR_QUIET" "WORKING_DIRECTORY;COMMAND_ERROR_IS_FATAL" "COMMAND" "${ARGV}")
    list(JOIN EX_COMMAND " " EX_COMMAND)
    message(STATUS "+ " ${EX_COMMAND})
    execute_process(
        ${ARGN}
        COMMAND_ERROR_IS_FATAL  ANY
        OUTPUT_QUIET ERROR_QUIET
    )
endfunction()

function(GitHub)
    cmake_parse_arguments(GH "" "USER;REPO;BRANCH;AS;PACK;FLAGS;DIR" "" "${ARGV}")
    message(STATUS "GITHUB " ${GH_USER}/${GH_REPO}/${GH_BRANCH})
    # download package from github
    if (NOT EXISTS "${CMAKE_BINARY_DIR}/_deps/${GH_USER}/${GH_REPO}/.git")
        Execute(
            COMMAND                 git clone --quiet --depth 1 --branch "${GH_BRANCH}" "https://github.com/${GH_USER}/${GH_REPO}" "${CMAKE_BINARY_DIR}/_deps/${GH_USER}/${GH_REPO}"
            WORKING_DIRECTORY       ${CMAKE_BINARY_DIR}
            COMMAND_ERROR_IS_FATAL  ANY
            OUTPUT_QUIET ERROR_QUIET
        )
    endif()
    # configure how this package is added to current project
    if(${GH_AS} STREQUAL "CMAKE SUBDIR")
        set(CMAKE_MESSAGE_LOG_LEVEL__ ${CMAKE_MESSAGE_LOG_LEVEL})
        set(CMAKE_MESSAGE_LOG_LEVEL ERROR)
        add_subdirectory(${CMAKE_BINARY_DIR}/_deps/${GH_USER}/${GH_REPO} EXCLUDE_FROM_ALL)
        set(CMAKE_MESSAGE_LOG_LEVEL ${CMAKE_MESSAGE_LOG_LEVEL__})
    elseif(${GH_AS} STREQUAL "CMAKE INSTALL" OR ${GH_AS} STREQUAL "CMAKE SCANLIB")
        # run the build command if it targeted directory don't exists
        if(NOT EXISTS ${CMAKE_BINARY_DIR}/_deps/${GH_USER}/${GH_REPO}/install)
            Execute(
                COMMAND                 cmake -S . -B build ${GH_FLAGS} -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/_deps/${GH_USER}/${GH_REPO}/install
                WORKING_DIRECTORY       ${CMAKE_BINARY_DIR}/_deps/${GH_USER}/${GH_REPO}
            )
            Execute(
                COMMAND                 cmake --build build -j
                WORKING_DIRECTORY       ${CMAKE_BINARY_DIR}/_deps/${GH_USER}/${GH_REPO}
            )
            Execute(
                COMMAND                 cmake --build build --target install
                WORKING_DIRECTORY       ${CMAKE_BINARY_DIR}/_deps/${GH_USER}/${GH_REPO}
            )
        endif()
        set(CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/_deps/${GH_USER}/${GH_REPO}/install")
        if("${GH_PACK}" STREQUAL "")
            set(GH_PACK ${GH_REPO})
        endif()
        if(${GH_AS} STREQUAL "CMAKE INSTALL")
            find_package(${GH_PACK} REQUIRED QUIET)
        elseif(${GH_AS} STREQUAL "CMAKE SCANLIB")
            include_directories(${CMAKE_BINARY_DIR}/_deps/${GH_USER}/${GH_REPO}/install/include)
            link_directories(${CMAKE_BINARY_DIR}/_deps/${GH_USER}/${GH_REPO}/install/lib)
        endif()
    elseif(${GH_AS} STREQUAL "AUTOMAKE INSTALL")
        # run the build command if it targeted directory don't exists
        if(NOT EXISTS ${CMAKE_BINARY_DIR}/_deps/${GH_USER}/${GH_REPO}/install)
            Execute(
                COMMAND                 ./autogen.sh --prefix ${CMAKE_BINARY_DIR}/_deps/${GH_USER}/${GH_REPO}/install
                WORKING_DIRECTORY       ${CMAKE_BINARY_DIR}/_deps/${GH_USER}/${GH_REPO}
            )
            Execute(
                COMMAND                 make install -j4
                WORKING_DIRECTORY       ${CMAKE_BINARY_DIR}/_deps/${GH_USER}/${GH_REPO}
            )
        endif()
        include_directories(${CMAKE_BINARY_DIR}/_deps/${GH_USER}/${GH_REPO}/install/include)
        link_directories(${CMAKE_BINARY_DIR}/_deps/${GH_USER}/${GH_REPO}/install/lib)
    else()
        message(FATAL_ERROR "UNKNOWN \"AUTOFIND OPTION\"")
    endif()
endfunction()

GitHub(
    USER    "ethereum"
    REPO    "evmc"
    BRANCH  "v11.0.1"
    AS      "CMAKE SUBDIR"
)
GitHub(
    USER    "fmtlib"
    REPO    "fmt"
    BRANCH  "10.2.1"
    AS      "CMAKE INSTALL"
    FLAGS   "-DBUILD_SHARED_LIBS=TRUE"
)
GitHub(
    USER    "chfast"
    REPO    "ethash"
    BRANCH  "v1.0.1"
    AS      "CMAKE INSTALL"
)
GitHub(
    USER    "chfast"
    REPO    "intx"
    BRANCH  "v0.10.1"
    AS      "CMAKE INSTALL"
)
GitHub(
    USER    "gflags"
    REPO    "gflags"
    BRANCH  "v2.2.2"
    AS      "CMAKE INSTALL"
    FLAGS   "-DBUILD_SHARED_LIBS=ON"
)
GitHub(
    USER    "google"
    REPO    "glog"
    BRANCH  "v0.7.0"
    AS      "CMAKE INSTALL"
    FLAGS   "-Dgflags_DIR=${gflags_DIR}"
)
GitHub(
    USER    "google"
    REPO    "googletest"
    BRANCH  "v1.14.0"
    PACK    "GTest"
    AS      "CMAKE SUBDIR"
)
GitHub(
    USER    "gperftools"
    REPO    "gperftools"
    BRANCH  "gperftools-2.15"
    AS      "CMAKE SCANLIB"
)
GitHub(
    USER    "jemalloc"
    REPO    "jemalloc"
    BRANCH  "5.3.0"
    AS      "AUTOMAKE INSTALL"
)

# -- Manage Compile Options w/ ASAN flag

if(NDEBUG)
    add_compile_options(
        -O3 -Werror -Wno-attributes
        -DNDEBUG=1
        -DPROJECT_VERSION=""
        -DEVMONE_X86_64_ARCH_LEVEL=2
    )
else()
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
    add_compile_options(
        -O0 -ggdb -Werror -Wno-attributes
        -DPROJECT_VERSION=""
        -DEVMONE_X86_64_ARCH_LEVEL=2
    )
endif()

# -- Process Main Library & Unit Tests

file(GLOB_RECURSE SRC lib/*.cpp lib/*.c)
file(GLOB_RECURSE INC lib/*.hpp lib/*.h)
file(GLOB_RECURSE SRC_TEST lib/*.test.cpp)
file(GLOB_RECURSE INC_TEST lib/*.test.hpp lib/*.test.h)

list(REMOVE_ITEM SRC ${SRC_TEST}) # remove *.test.cpp and *.test.hpp from target
list(REMOVE_ITEM INC ${INC_TEST})

foreach(F ${INC}) # copy headers
    file(RELATIVE_PATH R ${CMAKE_CURRENT_SOURCE_DIR}/lib ${F})
    configure_file(lib/${R} include/spectrum/${R} COPYONLY)
endforeach(F R)

add_library(spectrum STATIC ${SRC})

target_include_directories(
    spectrum
    PUBLIC  ${CMAKE_BINARY_DIR}/include
)
target_compile_features(
    spectrum
    PUBLIC cxx_std_20
)
target_link_libraries(
    spectrum
    PUBLIC  fmt::fmt evmc intx::intx glog::glog gflags
    PRIVATE ethash::keccak Threads::Threads jemalloc
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

message(STATUS "SPECTRUM LIBRARY")
foreach(F ${SRC})
    message(STATUS "+ " ${F})
endforeach(F R)

message(STATUS "UNIT TESTS")
enable_testing()
include(GoogleTest)
foreach(F ${SRC_TEST}) # unit tests
    file(RELATIVE_PATH R ${CMAKE_CURRENT_SOURCE_DIR} ${F})
    string(REPLACE "/" "-" R ${R})
    string(REPLACE ".cpp" "" R ${R})
    message(STATUS "+ " ${F})
    add_executable(${R} ${F})
    target_link_libraries(${R} GTest::gtest_main spectrum)
    set_target_properties(${R}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Testing
    )
    gtest_add_tests(TARGET ${R})
endforeach(F N)

unset(SRC_TEST)
unset(INC_TEST)
unset(INC)
unset(SRC)

# -- Process Executables

file(GLOB_RECURSE SRC bin/*.cpp)

message(STATUS "EXECUTABLES")
foreach(F ${SRC})
    message(STATUS "+ " ${F})
    file(RELATIVE_PATH R ${CMAKE_CURRENT_SOURCE_DIR}/bin ${F})
    string(REPLACE ".cpp" "" R ${R})
    add_executable(${R} ${F})
    target_link_libraries(${R} PRIVATE spectrum Threads::Threads)
endforeach(F R)

unset(SRC)
