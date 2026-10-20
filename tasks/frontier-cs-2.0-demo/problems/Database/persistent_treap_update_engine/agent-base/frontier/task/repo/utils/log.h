#pragma once

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LOG_OFF 0
#define LOG_FATAL 1
#define LOG_ERROR 2
#define LOG_WARN 3
#define LOG_INFO 4
#define LOG_DEBUG 5
#define LOG_TRACE 6
#define LOG_ALL 7

#ifndef LOG_LEVEL
#ifdef NDEBUG
#define LOG_LEVEL LOG_INFO
#else
#define LOG_LEVEL LOG_DEBUG
#endif
#endif

#define __log_message(level, file, line, func, fmt, args...) \
  fprintf(stderr, "[%s(%s:%d)] %s - " fmt "\n", func, file, line, level, ##args)

#if LOG_LEVEL >= LOG_FATAL
#define log_fatal(fmt, args...) \
  __log_message("FATAL", __FILE__, __LINE__, __func__, fmt, ##args)
#else
#define log_fatal(...) (0)
#endif

#if LOG_LEVEL >= LOG_ERROR
#define log_error(fmt, args...) \
  __log_message("ERROR", __FILE__, __LINE__, __func__, fmt, ##args)
#else
#define log_error(...) (0)
#endif

#if LOG_LEVEL >= LOG_WARN
#define log_warn(fmt, args...) \
  __log_message("WARN", __FILE__, __LINE__, __func__, fmt, ##args)
#else
#define log_warn(...) (0)
#endif

#if LOG_LEVEL >= LOG_INFO
#define log_info(fmt, args...) \
  __log_message("INFO", __FILE__, __LINE__, __func__, fmt, ##args)
#else
#define log_info(...) (0)
#endif

#if LOG_LEVEL >= LOG_DEBUG
#define log_debug(fmt, args...) \
  __log_message("DEBUG", __FILE__, __LINE__, __func__, fmt, ##args)
#else
#define log_debug(...) (0)
#endif

#if LOG_LEVEL >= LOG_TRACE
#define log_trace(fmt, args...) \
  __log_message("TRACE", __FILE__, __LINE__, __func__, fmt, ##args)
#else
#define log_trace(...) (0)
#endif

#ifdef __cplusplus
}
#endif
