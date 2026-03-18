#pragma once

#include <acl/acl.h>

#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short

#define TILELANG_CHECK(stmt)                                                   \
  do {                                                                         \
    aclError __err = (stmt);                                                   \
    if (__err != ACL_SUCCESS) {                                                \
      snprintf(error_buf, ERROR_BUF_SIZE, "%s:%d: aclError %d", __FILE__,      \
               __LINE__, (int)__err);                                          \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define TILELANG_CHECK_LAST_ERROR(kernel_name)                                 \
  do {                                                                         \
    auto __err = aclrtGetLastError(ACL_RT_THREAD_LEVEL);                       \
    if (__err) {                                                               \
      snprintf(error_buf, ERROR_BUF_SIZE, kernel_name ": aclError %d",         \
               (int)__err);                                                    \
      return -1;                                                               \
    }                                                                          \
  } while (0)
