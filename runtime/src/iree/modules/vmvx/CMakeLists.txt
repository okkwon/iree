# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(_VMVX_OPTIONAL_COPTS)
set(_VMVX_OPTIONAL_DEPS)

set(XNNPACK_DIR "" CACHE STRING "XNNPACK directory")
set(XNNPACK_BUILD_DIR "" CACHE STRING "XNNPACK build directory")

add_library(xnnpack STATIC IMPORTED)
set_target_properties(xnnpack PROPERTIES IMPORTED_LOCATION ${XNNPACK_BUILD_DIR}/libXNNPACK.a)
set_target_properties(xnnpack PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${XNNPACK_DIR}/include)

add_library(pthreadpool STATIC IMPORTED)
set_target_properties(pthreadpool PROPERTIES IMPORTED_LOCATION ${XNNPACK_BUILD_DIR}/pthreadpool/libpthreadpool.a)
set_target_properties(pthreadpool PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${XNNPACK_BUILD_DIR}/pthreadpool-source/include)

#add_library(cpuinfo STATIC IMPORTED)
#set_target_properties(cpuinfo PROPERTIES IMPORTED_LOCATION ${XNNPACK_BUILD_DIR}/cpuinfo/libcpuinfo.a)


iree_cc_library(
  NAME
    vmvx
  COPTS
    ${_VMVX_OPTIONAL_COPTS}
  HDRS
    "module.h"
  INCLUDES
    ${XNNPACK_BUILD_DIR}/pthreadpool-source/include
    ${XNNPACK_DIR}/include
  TEXTUAL_HDRS
    "exports.inl"
  SRCS
  "elementwise.c"
  "module.c"
  DEFINES
    "IREE_HAVE_VMVX_MODULE"
  DEPS
    iree::base
    iree::builtins::ukernel
    iree::base::internal::cpu
    iree::vm
    ${_VMVX_OPTIONAL_DEPS}
  PUBLIC
)

target_link_libraries(iree_modules_vmvx_vmvx PUBLIC xnnpack pthreadpool cpuinfo)
