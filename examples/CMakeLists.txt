cmake_minimum_required(VERSION 3.16)
project(ops_repo_examples LANGUAGES CXX)

set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH
    "CANN toolkit root containing include/ and lib64/")
set(OPS_REPO_SOC "ascend910b" CACHE STRING "SoC whose installed OPP is used by this example")
set_property(CACHE OPS_REPO_SOC PROPERTY STRINGS ascend910b ascend910_93 ascend950)
set(OPS_REPO_OPP_PATH "${CMAKE_CURRENT_LIST_DIR}/../.local/${OPS_REPO_SOC}/opp" CACHE PATH
    "Custom OPP installation root containing vendors/ops_repo/")

if(NOT ASCEND_CANN_PACKAGE_PATH)
    message(FATAL_ERROR
        "Set -DASCEND_CANN_PACKAGE_PATH=/path/to/ascend-toolkit/latest (or source CANN set_env.sh).")
endif()

get_filename_component(ASCEND_CANN_PACKAGE_PATH "${ASCEND_CANN_PACKAGE_PATH}" ABSOLUTE)
get_filename_component(OPS_REPO_OPP_PATH "${OPS_REPO_OPP_PATH}" ABSOLUTE)
set(_ops_repo_api "${OPS_REPO_OPP_PATH}/vendors/ops_repo/op_api")
set(_cann_include_paths
    "${ASCEND_CANN_PACKAGE_PATH}/include"
    "${ASCEND_CANN_PACKAGE_PATH}/include/nnopbase"
    "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
    "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/nnopbase"
    "${ASCEND_CANN_PACKAGE_PATH}/runtime/include")
set(_cann_library_paths
    "${ASCEND_CANN_PACKAGE_PATH}/lib64"
    "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64"
    "${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64")

find_path(ACL_INCLUDE_DIR NAMES acl/acl.h
    PATHS ${_cann_include_paths} NO_DEFAULT_PATH)
find_path(ACLNN_INCLUDE_DIR NAMES aclnn/acl_meta.h
    PATHS ${_cann_include_paths} NO_DEFAULT_PATH)
find_path(OPS_REPO_API_INCLUDE_DIR NAMES aclnn_ops_repo_add.h
    PATHS "${_ops_repo_api}/include" "${_ops_repo_api}/include/aclnnop"
    NO_DEFAULT_PATH)
find_library(ASCENDCL_LIBRARY NAMES ascendcl
    PATHS ${_cann_library_paths} NO_DEFAULT_PATH)
find_library(OPS_REPO_API_LIBRARY NAMES cust_opapi
    PATHS "${_ops_repo_api}/lib" "${_ops_repo_api}/lib64" NO_DEFAULT_PATH)
# aclCreateTensor and executor ownership APIs belong to the operator base
# library. Older toolkit layouts expose them through libopapi instead.
find_library(ACLNN_BASE_LIBRARY NAMES nnopbase opapi
    PATHS ${_cann_library_paths} NO_DEFAULT_PATH)

foreach(_dependency ACL_INCLUDE_DIR ACLNN_INCLUDE_DIR OPS_REPO_API_INCLUDE_DIR
        ASCENDCL_LIBRARY OPS_REPO_API_LIBRARY ACLNN_BASE_LIBRARY)
    if(NOT ${_dependency})
        message(FATAL_ERROR
            "Missing ${_dependency}. Check ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH} "
            "and install the ops_repo OPP under OPS_REPO_OPP_PATH=${OPS_REPO_OPP_PATH}.")
    endif()
endforeach()

add_executable(ops_repo_add ops_repo_add.cpp)
target_compile_features(ops_repo_add PRIVATE cxx_std_17)
set_target_properties(ops_repo_add PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(ops_repo_add PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(ops_repo_add PRIVATE
    "${ACL_INCLUDE_DIR}" "${ACLNN_INCLUDE_DIR}" "${OPS_REPO_API_INCLUDE_DIR}")
target_link_libraries(ops_repo_add PRIVATE
    "${OPS_REPO_API_LIBRARY}" "${ACLNN_BASE_LIBRARY}" "${ASCENDCL_LIBRARY}")

get_filename_component(_custom_lib_dir "${OPS_REPO_API_LIBRARY}" DIRECTORY)
get_filename_component(_base_lib_dir "${ACLNN_BASE_LIBRARY}" DIRECTORY)
get_filename_component(_runtime_lib_dir "${ASCENDCL_LIBRARY}" DIRECTORY)
set_target_properties(ops_repo_add PROPERTIES
    BUILD_RPATH "${_custom_lib_dir};${_base_lib_dir};${_runtime_lib_dir}"
    INSTALL_RPATH "${_custom_lib_dir};${_base_lib_dir};${_runtime_lib_dir}")
# RPATH locates shared libraries; the vendor set_env.bash must still be sourced
# at runtime so CANN can discover the custom operator metadata and kernels.
