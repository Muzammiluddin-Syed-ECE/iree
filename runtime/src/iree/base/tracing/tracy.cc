// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/tracing.h"

// Textually include the Tracy implementation.
// We do this here instead of relying on an external build target so that we can
// ensure our configuration specified in tracing.h is picked up.
#if IREE_TRACING_FEATURES != 0
#include "TracyClient.cpp"
#endif  // IREE_TRACING_FEATURES

#if defined(TRACY_ENABLE) && IREE_TRACING_EXPERIMENTAL_CONTEXT_API
// HACK: tracy doesn't let us at this but we need it in order to create new
// queue contexts. It's an implementation detail we have to take a dependency on
// because tracy does not have an API for what we're doing (yet).
namespace tracy {
moodycamel::ConcurrentQueue<QueueItem>& GetQueue();
}  // namespace tracy
#endif  // TRACY_ENABLE && IREE_TRACING_EXPERIMENTAL_CONTEXT_API

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#if defined(TRACY_ENABLE) && defined(IREE_PLATFORM_WINDOWS)
static HANDLE iree_dbghelp_mutex;
void IREEDbgHelpInit(void) {
  iree_dbghelp_mutex = CreateMutex(NULL, FALSE, NULL);
}
void IREEDbgHelpLock(void) {
  WaitForSingleObject(iree_dbghelp_mutex, INFINITE);
}
void IREEDbgHelpUnlock(void) { ReleaseMutex(iree_dbghelp_mutex); }
#endif  // TRACY_ENABLE && IREE_PLATFORM_WINDOWS

#if IREE_TRACING_FEATURES != 0

typedef struct iree_tracing_source_file_t {
  uint8_t* filename;
  size_t filename_length;
  uint8_t* content;
  size_t content_length;
} iree_tracing_source_file_t;

// Global registry of published source files allocated using the Tracy allocator
// and live for the entire lifetime of the program as Tracy will request the
// contents long past tear-down.
typedef struct iree_tracing_source_file_storage_t {
  tracy::TracyMutex mutex;
  iree_host_size_t capacity;
  iree_host_size_t count;
  iree_tracing_source_file_t** files;
} iree_tracing_source_file_storage_t;
static iree_tracing_source_file_storage_t iree_tracing_source_file_storage;

static char* iree_tracing_tracy_source_file_callback(void* user_data,
                                                     const char* filename,
                                                     size_t& out_size) {
  iree_tracing_source_file_storage_t* storage =
      (iree_tracing_source_file_storage_t*)user_data;

  const iree_host_size_t filename_length = strlen(filename);
  char* content_copy = NULL;
  iree_host_size_t content_length = 0;

  storage->mutex.lock();

  for (iree_host_size_t i = 0; i < storage->count; ++i) {
    iree_tracing_source_file_t* source_file = storage->files[i];
    if (filename_length != source_file->filename_length) continue;
    // NOTE: no case-insensitive/fuzzy comparison (yet). The paths are
    // generated by the compiler in the same place and they should always line
    // up but if we start embedding arbitrary user files we may need to
    // normalize paths.
    if (memcmp(filename, source_file->filename, filename_length) == 0) {
      content_copy =
          (char*)tracy::tracy_malloc_fast(source_file->content_length);
      memcpy(content_copy, source_file->content, source_file->content_length);
      content_length = source_file->content_length;
      break;
    }
  }

  storage->mutex.unlock();

  out_size = content_length;
  return content_copy;
}

void iree_tracing_tracy_initialize() {
#ifdef TRACY_MANUAL_LIFETIME
  tracy::StartupProfiler();
#endif  // TRACY_MANUAL_LIFETIME
  // Register a single source provider callback with Tracy. Tracy only supports
  // one at a time and the callback must remain valid until program exit.
  tracy::Profiler::SourceCallbackRegister(
      iree_tracing_tracy_source_file_callback,
      &iree_tracing_source_file_storage);
}

void iree_tracing_tracy_deinitialize() {
#if defined(IREE_PLATFORM_APPLE)
  // Synchronously shut down the profiler service.
  // This is required on some platforms to support TRACY_NO_EXIT=1 such as
  // MacOS and iOS. It should be harmless on other platforms as it returns
  // quickly if TRACY_NO_EXIT=1 is not set.
  // See: https://github.com/wolfpld/tracy/issues/8
  tracy::GetProfiler().RequestShutdown();
  while (!tracy::GetProfiler().HasShutdownFinished()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
#endif  // IREE_PLATFORM_*
#ifdef TRACY_MANUAL_LIFETIME
  tracy::ShutdownProfiler();
#endif  // TRACY_MANUAL_LIFETIME
}

void iree_tracing_publish_source_file(const void* filename,
                                      size_t filename_length,
                                      const void* content,
                                      size_t content_length) {
  iree_tracing_source_file_storage_t* storage =
      &iree_tracing_source_file_storage;

  // NOTE: this does not currently check to see whether the file has already
  // been published. We could but in most valid usage we don't need to.

  // Allocate storage for the file - we do this as a single alloc of the entry
  // with the filename and content tacked on.
  size_t total_size =
      sizeof(iree_tracing_source_file_t) + filename_length + content_length;
  uint8_t* entry_ptr = (uint8_t*)tracy::tracy_malloc_fast(total_size);
  iree_tracing_source_file_t* source_file =
      (iree_tracing_source_file_t*)entry_ptr;
  source_file->filename = entry_ptr + sizeof(*source_file);
  source_file->filename_length = filename_length;
  memcpy(source_file->filename, filename, filename_length);
  source_file->content = source_file->filename + filename_length;
  source_file->content_length = content_length;
  memcpy(source_file->content, content, content_length);

  storage->mutex.lock();

  // Grow capacity of the storage index if needed.
  if (storage->count + 1 >= storage->capacity) {
    storage->capacity = std::max((iree_host_size_t)32, storage->capacity * 2);
    storage->files = (iree_tracing_source_file_t**)tracy::tracy_realloc(
        storage->files, storage->capacity * sizeof(*storage->files));
  }

  // Append the file.
  storage->files[storage->count++] = source_file;

  storage->mutex.unlock();
}

iree_zone_id_t iree_tracing_zone_begin_impl(
    const iree_tracing_location_t* src_loc, const char* name,
    size_t name_length) {
  const iree_zone_id_t zone_id = tracy::GetProfiler().GetNextZoneId();

#ifndef TRACY_NO_VERIFY
  {
    TracyQueuePrepareC(tracy::QueueType::ZoneValidation);
    tracy::MemWrite(&item->zoneValidation.id, zone_id);
    TracyQueueCommitC(zoneValidationThread);
  }
#endif  // TRACY_NO_VERIFY

  {
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_CALLSTACKS
    TracyQueuePrepareC(tracy::QueueType::ZoneBeginCallstack);
#else
    TracyQueuePrepareC(tracy::QueueType::ZoneBegin);
#endif  // IREE_TRACING_FEATURE_INSTRUMENTATION_CALLSTACKS
    tracy::MemWrite(&item->zoneBegin.time, tracy::Profiler::GetTime());
    tracy::MemWrite(&item->zoneBegin.srcloc,
                    reinterpret_cast<uint64_t>(src_loc));
    TracyQueueCommitC(zoneBeginThread);
  }

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_CALLSTACKS
  tracy::GetProfiler().SendCallstack(IREE_TRACING_MAX_CALLSTACK_DEPTH);
#endif  // IREE_TRACING_FEATURE_INSTRUMENTATION_CALLSTACKS

  if (name_length) {
#ifndef TRACY_NO_VERIFY
    {
      TracyQueuePrepareC(tracy::QueueType::ZoneValidation);
      tracy::MemWrite(&item->zoneValidation.id, zone_id);
      TracyQueueCommitC(zoneValidationThread);
    }
#endif  // TRACY_NO_VERIFY
    auto name_ptr = reinterpret_cast<char*>(tracy::tracy_malloc(name_length));
    memcpy(name_ptr, name, name_length);
    TracyQueuePrepareC(tracy::QueueType::ZoneName);
    tracy::MemWrite(&item->zoneTextFat.text,
                    reinterpret_cast<uint64_t>(name_ptr));
    tracy::MemWrite(&item->zoneTextFat.size,
                    static_cast<uint64_t>(name_length));
    TracyQueueCommitC(zoneTextFatThread);
  }

  return zone_id;
}

iree_zone_id_t iree_tracing_zone_begin_external_impl(
    const char* file_name, size_t file_name_length, uint32_t line,
    const char* function_name, size_t function_name_length, const char* name,
    size_t name_length) {
  uint64_t src_loc = tracy::Profiler::AllocSourceLocation(
      line, file_name, file_name_length, function_name, function_name_length,
      name, name_length);

  const iree_zone_id_t zone_id = tracy::GetProfiler().GetNextZoneId();

#ifndef TRACY_NO_VERIFY
  {
    TracyQueuePrepareC(tracy::QueueType::ZoneValidation);
    tracy::MemWrite(&item->zoneValidation.id, zone_id);
    TracyQueueCommitC(zoneValidationThread);
  }
#endif  // TRACY_NO_VERIFY

  {
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_CALLSTACKS
    TracyQueuePrepareC(tracy::QueueType::ZoneBeginAllocSrcLocCallstack);
#else
    TracyQueuePrepareC(tracy::QueueType::ZoneBeginAllocSrcLoc);
#endif  // IREE_TRACING_FEATURE_INSTRUMENTATION_CALLSTACKS
    tracy::MemWrite(&item->zoneBegin.time, tracy::Profiler::GetTime());
    tracy::MemWrite(&item->zoneBegin.srcloc, src_loc);
    TracyQueueCommitC(zoneBeginThread);
  }

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_CALLSTACKS
  tracy::GetProfiler().SendCallstack(IREE_TRACING_MAX_CALLSTACK_DEPTH);
#endif  // IREE_TRACING_FEATURE_INSTRUMENTATION_CALLSTACKS

  return zone_id;
}

void iree_tracing_zone_end(iree_zone_id_t zone_id) {
  ___tracy_emit_zone_end(iree_tracing_make_zone_ctx(zone_id));
}

void iree_tracing_set_plot_type_impl(const char* name_literal,
                                     uint8_t plot_type, bool step, bool fill,
                                     uint32_t color) {
  tracy::Profiler::ConfigurePlot(name_literal,
                                 static_cast<tracy::PlotFormatType>(plot_type),
                                 step, fill, color);
}

void iree_tracing_plot_value_i64_impl(const char* name_literal, int64_t value) {
  tracy::Profiler::PlotData(name_literal, value);
}

void iree_tracing_plot_value_f32_impl(const char* name_literal, float value) {
  tracy::Profiler::PlotData(name_literal, value);
}

void iree_tracing_plot_value_f64_impl(const char* name_literal, double value) {
  tracy::Profiler::PlotData(name_literal, value);
}

void iree_tracing_mutex_announce(const iree_tracing_location_t* src_loc,
                                 uint32_t* out_lock_id) {
  uint32_t lock_id =
      tracy::GetLockCounter().fetch_add(1, std::memory_order_relaxed);
  assert(lock_id != std::numeric_limits<uint32_t>::max());
  *out_lock_id = lock_id;

  auto item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::LockAnnounce);
  tracy::MemWrite(&item->lockAnnounce.id, lock_id);
  tracy::MemWrite(&item->lockAnnounce.time, tracy::Profiler::GetTime());
  tracy::MemWrite(&item->lockAnnounce.lckloc,
                  reinterpret_cast<uint64_t>(src_loc));
  tracy::MemWrite(&item->lockAnnounce.type, tracy::LockType::Lockable);
  tracy::Profiler::QueueSerialFinish();
}

void iree_tracing_mutex_terminate(uint32_t lock_id) {
  auto item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::LockTerminate);
  tracy::MemWrite(&item->lockTerminate.id, lock_id);
  tracy::MemWrite(&item->lockTerminate.time, tracy::Profiler::GetTime());
  tracy::Profiler::QueueSerialFinish();
}

void iree_tracing_mutex_before_lock(uint32_t lock_id) {
  auto item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::LockWait);
  tracy::MemWrite(&item->lockWait.thread, tracy::GetThreadHandle());
  tracy::MemWrite(&item->lockWait.id, lock_id);
  tracy::MemWrite(&item->lockWait.time, tracy::Profiler::GetTime());
  tracy::Profiler::QueueSerialFinish();
}

void iree_tracing_mutex_after_lock(uint32_t lock_id) {
  auto item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::LockObtain);
  tracy::MemWrite(&item->lockObtain.thread, tracy::GetThreadHandle());
  tracy::MemWrite(&item->lockObtain.id, lock_id);
  tracy::MemWrite(&item->lockObtain.time, tracy::Profiler::GetTime());
  tracy::Profiler::QueueSerialFinish();
}

void iree_tracing_mutex_after_try_lock(uint32_t lock_id, bool was_acquired) {
  if (was_acquired) {
    iree_tracing_mutex_after_lock(lock_id);
  }
}

void iree_tracing_mutex_after_unlock(uint32_t lock_id) {
  auto item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::LockRelease);
  tracy::MemWrite(&item->lockReleaseShared.thread, tracy::GetThreadHandle());
  tracy::MemWrite(&item->lockRelease.id, lock_id);
  tracy::MemWrite(&item->lockRelease.time, tracy::Profiler::GetTime());
  tracy::Profiler::QueueSerialFinish();
}

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

int64_t iree_tracing_time(void) { return tracy::Profiler::GetTime(); }

int64_t iree_tracing_frequency(void) { return tracy::GetFrequencyQpc(); }

uint8_t iree_tracing_gpu_context_allocate(iree_tracing_gpu_context_type_t type,
                                          const char* name, size_t name_length,
                                          bool is_calibrated,
                                          uint64_t cpu_timestamp,
                                          uint64_t gpu_timestamp,
                                          float timestamp_period) {
  // Allocate the process-unique GPU context ID. There's a max of 255 available;
  // if we are recreating devices a lot we may exceed that. Don't do that, or
  // wrap around and get weird (but probably still usable) numbers.
  uint8_t context_id =
      tracy::GetGpuCtxCounter().fetch_add(1, std::memory_order_relaxed);
  if (context_id >= 255) {
    context_id %= 255;
  }

  uint8_t context_flags = 0;
  if (is_calibrated) {
    // Tell tracy we'll be passing calibrated timestamps and not to mess with
    // the times. We'll periodically send GpuCalibration events in case the
    // times drift.
    context_flags |= tracy::GpuContextCalibration;
  }
  {
    auto* item = tracy::Profiler::QueueSerial();
    tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuNewContext);
    tracy::MemWrite(&item->gpuNewContext.cpuTime, cpu_timestamp);
    tracy::MemWrite(&item->gpuNewContext.gpuTime, gpu_timestamp);
    memset(&item->gpuNewContext.thread, 0, sizeof(item->gpuNewContext.thread));
    tracy::MemWrite(&item->gpuNewContext.period, timestamp_period);
    tracy::MemWrite(&item->gpuNewContext.context, context_id);
    tracy::MemWrite(&item->gpuNewContext.flags, context_flags);
    tracy::MemWrite(&item->gpuNewContext.type, (tracy::GpuContextType)type);
    tracy::Profiler::QueueSerialFinish();
  }

  // Send the name of the context along.
  // NOTE: Tracy will unconditionally free the name so we must clone it here.
  // Since internally Tracy will use its own rpmalloc implementation we must
  // make sure we allocate from the same source.
  char* cloned_name = (char*)tracy::tracy_malloc(name_length);
  memcpy(cloned_name, name, name_length);
  {
    auto* item = tracy::Profiler::QueueSerial();
    tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuContextName);
    tracy::MemWrite(&item->gpuContextNameFat.context, context_id);
    tracy::MemWrite(&item->gpuContextNameFat.ptr, (uint64_t)cloned_name);
    tracy::MemWrite(&item->gpuContextNameFat.size, name_length);
    tracy::Profiler::QueueSerialFinish();
  }

  return context_id;
}

void iree_tracing_gpu_context_calibrate(uint8_t context_id, int64_t cpu_delta,
                                        int64_t cpu_timestamp,
                                        int64_t gpu_timestamp) {
  auto* item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuCalibration);
  tracy::MemWrite(&item->gpuCalibration.gpuTime, gpu_timestamp);
  tracy::MemWrite(&item->gpuCalibration.cpuTime, cpu_timestamp);
  tracy::MemWrite(&item->gpuCalibration.cpuDelta, cpu_delta);
  tracy::MemWrite(&item->gpuCalibration.context, context_id);
  tracy::Profiler::QueueSerialFinish();
}

void iree_tracing_gpu_zone_begin(uint8_t context_id, uint16_t query_id,
                                 const iree_tracing_location_t* src_loc) {
  auto* item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuZoneBeginSerial);
  tracy::MemWrite(&item->gpuZoneBegin.cpuTime, tracy::Profiler::GetTime());
  tracy::MemWrite(&item->gpuZoneBegin.srcloc, (uint64_t)src_loc);
  tracy::MemWrite(&item->gpuZoneBegin.thread, tracy::GetThreadHandle());
  tracy::MemWrite(&item->gpuZoneBegin.queryId, query_id);
  tracy::MemWrite(&item->gpuZoneBegin.context, context_id);
  tracy::Profiler::QueueSerialFinish();
}

void iree_tracing_gpu_zone_begin_external(
    uint8_t context_id, uint16_t query_id, const char* file_name,
    size_t file_name_length, uint32_t line, const char* function_name,
    size_t function_name_length, const char* name, size_t name_length) {
  const auto src_loc = tracy::Profiler::AllocSourceLocation(
      line, file_name, file_name_length, function_name, function_name_length,
      name, name_length);
  auto* item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type,
                  tracy::QueueType::GpuZoneBeginAllocSrcLocSerial);
  tracy::MemWrite(&item->gpuZoneBegin.cpuTime, tracy::Profiler::GetTime());
  tracy::MemWrite(&item->gpuZoneBegin.srcloc, (uint64_t)src_loc);
  tracy::MemWrite(&item->gpuZoneBegin.thread, tracy::GetThreadHandle());
  tracy::MemWrite(&item->gpuZoneBegin.queryId, query_id);
  tracy::MemWrite(&item->gpuZoneBegin.context, context_id);
  tracy::Profiler::QueueSerialFinish();
}

void iree_tracing_gpu_zone_end(uint8_t context_id, uint16_t query_id) {
  auto* item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuZoneEndSerial);
  tracy::MemWrite(&item->gpuZoneEnd.cpuTime, tracy::Profiler::GetTime());
  tracy::MemWrite(&item->gpuZoneEnd.thread, tracy::GetThreadHandle());
  tracy::MemWrite(&item->gpuZoneEnd.queryId, query_id);
  tracy::MemWrite(&item->gpuZoneEnd.context, context_id);
  tracy::Profiler::QueueSerialFinish();
}

void iree_tracing_gpu_zone_notify(uint8_t context_id, uint16_t query_id,
                                  int64_t gpu_timestamp) {
  auto* item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuTime);
  tracy::MemWrite(&item->gpuTime.gpuTime, gpu_timestamp);
  tracy::MemWrite(&item->gpuTime.queryId, query_id);
  tracy::MemWrite(&item->gpuTime.context, context_id);
  tracy::Profiler::QueueSerialFinish();
}

#endif  // IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
void* iree_tracing_obscure_ptr(void* ptr) { return ptr; }
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

//===----------------------------------------------------------------------===//
// Experimental Tracing Interop API
//===----------------------------------------------------------------------===//

#if IREE_TRACING_EXPERIMENTAL_CONTEXT_API

struct iree_tracing_context_t {
  inline static std::atomic<uint32_t> next_tracing_thread_id{0x80000000u};
  tracy::moodycamel::ProducerToken token_detail;
  tracy::ProducerWrapper token;
  uint32_t thread_id = 0;
  iree_tracing_context_t()
      : token_detail(tracy::GetQueue()),
        token({tracy::GetQueue().get_explicit_producer(token_detail)}),
        thread_id(iree_tracing_context_t::next_tracing_thread_id++) {
    token.ptr->threadId = thread_id;
  }
};

#define IREE_TRACING_CONTEXT_BEGIN_WRITE(context, queue_type)             \
  tracy::moodycamel::ConcurrentQueueDefaultTraits::index_t __magic;       \
  tracy::moodycamel::ConcurrentQueue<tracy::QueueItem>::ExplicitProducer* \
      __token = (context)->token.ptr;                                     \
  auto& __tail = __token->get_tail_index();                               \
  auto item = __token->enqueue_begin(__magic);                            \
  tracy::MemWrite(&item->hdr.type, (queue_type));

#define IREE_TRACING_CONTEXT_END_WRITE(context) \
  __tail.store(__magic + 1, std::memory_order_release);

iree_tracing_context_t* iree_tracing_context_allocate(
    const char* name, iree_host_size_t name_length) {
  iree_tracing_context_t* context = new iree_tracing_context_t();

  // TODO(benvanik): upstream a tracy::Profiler::SetThreadNameWithHint that
  // only updates the GetThreadNameData() linked list with a new entry. Today
  // there's no way to set the thread name explicitly.

  return context;
}

void iree_tracing_context_free(iree_tracing_context_t* context) {
  if (context) delete context;
}

void iree_tracing_context_calibrate_executor(
    iree_tracing_context_t* context, iree_tracing_executor_id_t executor_id,
    int64_t cpu_delta, uint64_t host_timestamp, uint64_t executor_timestamp) {
  IREE_TRACING_CONTEXT_BEGIN_WRITE(context, tracy::QueueType::GpuCalibration);
  tracy::MemWrite(&item->gpuCalibration.gpuTime, executor_timestamp);
  tracy::MemWrite(&item->gpuCalibration.cpuTime, host_timestamp);
  tracy::MemWrite(&item->gpuCalibration.cpuDelta, cpu_delta);
  tracy::MemWrite(&item->gpuCalibration.context, executor_id);
  IREE_TRACING_CONTEXT_END_WRITE(context);
}

void iree_tracing_context_zone_begin(iree_tracing_context_t* context,
                                     uint64_t timestamp,
                                     const iree_tracing_location_t* src_loc) {
  IREE_TRACING_CONTEXT_BEGIN_WRITE(context, tracy::QueueType::ZoneBegin);
  tracy::MemWrite(&item->zoneBegin.time, timestamp);
  tracy::MemWrite(&item->zoneBegin.srcloc, reinterpret_cast<uint64_t>(src_loc));
  IREE_TRACING_CONTEXT_END_WRITE(context);
}

void iree_tracing_context_zone_end(iree_tracing_context_t* context,
                                   uint64_t timestamp) {
  IREE_TRACING_CONTEXT_BEGIN_WRITE(context, tracy::QueueType::ZoneEnd);
  tracy::MemWrite(&item->zoneEnd.time, timestamp);
  IREE_TRACING_CONTEXT_END_WRITE(context);
}

void iree_tracing_context_zone_value_i64(iree_tracing_context_t* context,
                                         uint64_t value) {
  IREE_TRACING_CONTEXT_BEGIN_WRITE(context, tracy::QueueType::ZoneValue);
  tracy::MemWrite(&item->zoneValue.value, value);
  IREE_TRACING_CONTEXT_END_WRITE(context);
}

void iree_tracing_context_zone_value_text_literal(
    iree_tracing_context_t* context, const char* value) {
  // NOTE: no literal tracing support, have to use the slow path.
  iree_tracing_context_zone_value_text_dynamic(context, value, strlen(value));
}

void iree_tracing_context_zone_value_text_dynamic(
    iree_tracing_context_t* context, const char* value,
    iree_host_size_t value_length) {
  auto ptr = (char*)tracy::tracy_malloc(value_length);
  memcpy(ptr, value, value_length);
  IREE_TRACING_CONTEXT_BEGIN_WRITE(context, tracy::QueueType::ZoneText);
  tracy::MemWrite(&item->zoneTextFat.text, (uint64_t)ptr);
  tracy::MemWrite(&item->zoneTextFat.size, (uint16_t)value_length);
  IREE_TRACING_CONTEXT_END_WRITE(context);
}

// TODO(benvanik): figure out why serial recording works with GPU zones and
// thread-local recording doesn't (sometimes?). May be timing related.
#define IREE_TRACING_CONTEXT_SERIAL_FALLBACK 1

void iree_tracing_context_execution_zone_begin(
    iree_tracing_context_t* context, uint64_t timestamp,
    const iree_tracing_location_t* src_loc,
    iree_tracing_executor_id_t executor_id, iree_tracing_query_id_t query_id) {
#if IREE_TRACING_CONTEXT_SERIAL_FALLBACK
  auto* item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuZoneBeginSerial);
  tracy::MemWrite(&item->gpuZoneBegin.cpuTime, timestamp);
  tracy::MemWrite(&item->gpuZoneBegin.srcloc, src_loc);
  tracy::MemWrite(&item->gpuZoneBegin.thread, context->thread_id);
  tracy::MemWrite(&item->gpuZoneBegin.queryId, query_id);
  tracy::MemWrite(&item->gpuZoneBegin.context, executor_id);
  tracy::Profiler::QueueSerialFinish();
#else
  IREE_TRACING_CONTEXT_BEGIN_WRITE(context, tracy::QueueType::GpuZoneBegin);
  tracy::MemWrite(&item->gpuZoneBegin.cpuTime, timestamp);
  tracy::MemWrite(&item->gpuZoneBegin.thread, context->thread_id);
  tracy::MemWrite(&item->gpuZoneBegin.queryId, query_id);
  tracy::MemWrite(&item->gpuZoneBegin.context, executor_id);
  tracy::MemWrite(&item->gpuZoneBegin.srcloc, src_loc);
  IREE_TRACING_CONTEXT_END_WRITE(context);
#endif  // IREE_TRACING_CONTEXT_SERIAL_FALLBACK
}

void iree_tracing_context_execution_zone_end(
    iree_tracing_context_t* context, uint64_t timestamp,
    iree_tracing_executor_id_t executor_id, iree_tracing_query_id_t query_id) {
#if IREE_TRACING_CONTEXT_SERIAL_FALLBACK
  auto* item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuZoneEndSerial);
  tracy::MemWrite(&item->gpuZoneEnd.cpuTime, timestamp);
  tracy::MemWrite(&item->gpuZoneEnd.thread, context->thread_id);
  tracy::MemWrite(&item->gpuZoneEnd.queryId, query_id);
  tracy::MemWrite(&item->gpuZoneEnd.context, executor_id);
  tracy::Profiler::QueueSerialFinish();
#else
  IREE_TRACING_CONTEXT_BEGIN_WRITE(context, tracy::QueueType::GpuZoneEnd);
  tracy::MemWrite(&item->gpuZoneEnd.cpuTime, timestamp);
  tracy::MemWrite(&item->gpuZoneEnd.thread, context->thread_id);
  tracy::MemWrite(&item->gpuZoneEnd.queryId, query_id);
  tracy::MemWrite(&item->gpuZoneEnd.context, executor_id);
  IREE_TRACING_CONTEXT_END_WRITE(context);
#endif  // IREE_TRACING_CONTEXT_SERIAL_FALLBACK
}

void iree_tracing_context_execution_zone_notify(
    iree_tracing_context_t* context, iree_tracing_executor_id_t executor_id,
    iree_tracing_query_id_t query_id, uint64_t query_timestamp) {
#if IREE_TRACING_CONTEXT_SERIAL_FALLBACK
  iree_tracing_gpu_zone_notify(executor_id, query_id, query_timestamp);
  auto* item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuTime);
  tracy::MemWrite(&item->gpuTime.gpuTime, query_timestamp);
  tracy::MemWrite(&item->gpuTime.queryId, query_id);
  tracy::MemWrite(&item->gpuTime.context, executor_id);
  tracy::Profiler::QueueSerialFinish();
#else
  IREE_TRACING_CONTEXT_BEGIN_WRITE(context, tracy::QueueType::GpuTime);
  tracy::MemWrite(&item->gpuTime.gpuTime, query_timestamp);
  tracy::MemWrite(&item->gpuTime.queryId, query_id);
  tracy::MemWrite(&item->gpuTime.context, executor_id);
  IREE_TRACING_CONTEXT_END_WRITE(context);
#endif  // IREE_TRACING_CONTEXT_SERIAL_FALLBACK
}

void iree_tracing_context_memory_alloc(iree_tracing_context_t* context,
                                       uint64_t timestamp, const char* pool,
                                       uint64_t ptr, uint64_t size) {
  // TODO(benvanik): add a thread override to MemAllocNamed - it does shady
  // things with m_memNamePayload that we can't easily replicate outside of the
  // tracy implementation.
}

void iree_tracing_context_memory_free(iree_tracing_context_t* context,
                                      uint64_t timestamp, const char* pool,
                                      uint64_t ptr) {
  // TODO(benvanik): add a thread override to MemFreeNamed- it does shady
  // things with m_memNamePayload that we can't easily replicate outside of the
  // tracy implementation.
}

void iree_tracing_context_message_literal(iree_tracing_context_t* context,
                                          uint64_t timestamp,
                                          const char* value) {
  IREE_TRACING_CONTEXT_BEGIN_WRITE(context, tracy::QueueType::MessageLiteral);
  tracy::MemWrite(&item->messageLiteral.time, timestamp);
  tracy::MemWrite(&item->messageLiteral.text, (uint64_t)value);
  IREE_TRACING_CONTEXT_END_WRITE(context);
}

void iree_tracing_context_message_dynamic(iree_tracing_context_t* context,
                                          uint64_t timestamp, const char* value,
                                          iree_host_size_t value_length) {
  auto ptr = (char*)tracy::tracy_malloc(value_length);
  memcpy(ptr, value, value_length);
  IREE_TRACING_CONTEXT_BEGIN_WRITE(context, tracy::QueueType::Message);
  tracy::MemWrite(&item->messageFat.time, timestamp);
  tracy::MemWrite(&item->messageFat.text, (uint64_t)ptr);
  tracy::MemWrite(&item->messageFat.size, (uint16_t)value_length);
  IREE_TRACING_CONTEXT_END_WRITE(context);
}

void iree_tracing_context_plot_config(iree_tracing_context_t* context,
                                      const char* name_literal, uint8_t type,
                                      bool step, bool fill, uint32_t color) {
  IREE_TRACING_CONTEXT_BEGIN_WRITE(context, tracy::QueueType::PlotConfig);
  tracy::MemWrite(&item->plotConfig.name, (uint64_t)name_literal);
  tracy::MemWrite(&item->plotConfig.type, (uint8_t)type);
  tracy::MemWrite(&item->plotConfig.step, (uint8_t)step);
  tracy::MemWrite(&item->plotConfig.fill, (uint8_t)fill);
  tracy::MemWrite(&item->plotConfig.color, color);
  IREE_TRACING_CONTEXT_END_WRITE(context);
}

void iree_tracing_context_plot_value_i64(iree_tracing_context_t* context,
                                         uint64_t timestamp,
                                         const char* plot_name, int64_t value) {
  IREE_TRACING_CONTEXT_BEGIN_WRITE(context, tracy::QueueType::PlotDataInt);
  tracy::MemWrite(&item->plotDataInt.name, (uint64_t)plot_name);
  tracy::MemWrite(&item->plotDataInt.time, timestamp);
  tracy::MemWrite(&item->plotDataInt.val, value);
  IREE_TRACING_CONTEXT_END_WRITE(context);
}

#endif  // IREE_TRACING_EXPERIMENTAL_CONTEXT_API

#endif  // IREE_TRACING_FEATURES

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
