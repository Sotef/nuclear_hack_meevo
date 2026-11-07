// hashshim.cpp — provides std::__ndk1::__hash_memory removed in newer libc++.
// sherpa-onnx v2.0.1 termux binaries were built against an older libc++ that
// exported this symbol; NDK r27c libc++_shared.so no longer does. LD_PRELOAD
// this shim so the executable links. Deterministic FNV-1a is sufficient: all
// in-process hash calls resolve to this single definition.
#include <cstddef>
#include <cstdint>

namespace std {
namespace __ndk1 {
size_t __hash_memory(const void* ptr, size_t size) noexcept {
  const unsigned char* p = static_cast<const unsigned char*>(ptr);
  size_t h = static_cast<size_t>(1469598103934665603ULL);
  for (size_t i = 0; i < size; ++i) {
    h ^= static_cast<size_t>(p[i]);
    h *= static_cast<size_t>(1099511628211ULL);
  }
  return h;
}
}  // namespace __ndk1
}  // namespace std
