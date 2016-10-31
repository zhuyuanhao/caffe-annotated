#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#include "caffe/common.hpp"

namespace caffe {

// CaffeMallocHost和CaffeFreeHost组成一组CPU内存分配和释放的操作，当存在GPU数据时，这些内存可用于
// 与GPU内存的数据交换，GPU访问cuda pinned memory时速度更快。分配时通过Caffe::mode()
// 判断用pageable memory（无GPU）或cuda pinned memory（有GPU），并设置参数use_cuda。释放时
// 通过use_cuda确定是用CPU还是GPU的内存释放函数

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
// ptr是void**类型，size是数据的字节数
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size)); // 注意cudaMallocHost函数的传入参数是void**类型
    *use_cuda = true;
    return;
  }
#endif
  *ptr = malloc(size);
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
  free(ptr);
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
 // SyncedMemory为程序在CPU或GPU运行时提供同样内容的数据，程序在CPU上运行时，
 // 获取CPU内存地址，在GPU上运行时，获取GPU内存地址，SyncedMemory内部会处理数据的
 // 分配、释放和同步，保证它们的数据内容都是一致的
 // 数据的分配是按需分配，同步也是按需同步，只有当数据在一端(cpu/gpu)修改后，在另一端访问时才需要同步数据
class SyncedMemory {
 public:
  // 初始化函数都不会分配内存，head状态为UNINITIALIZED,只有在调用了一次 [set/mutable_]cpu/gpu_data()后cpu/gpu内存才会申请
  SyncedMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  ~SyncedMemory();

  // cpu/gpu_data()返回数据的const指针，mutable_cpu/gpu_data()返回可修改的指针
  // set_cpu/gpu_data()使用外部传入的内存块，同时释放自己申请的内存块，它通过own_cpu/gpu_data_判断
  // 持有的内存块是否是自己申请的
  // set/mutable_cpu/gpu_data()都会修改内存，head的状态设置为HEAD_AT_CPU/HEAD_AT_GPU
  // [mutable_]cpu/gpu_data()内部都是调用了to_cpu/gpu()函数，to_cpu/gpu()在head为UNINITIAIZED或另一端时才需要
  // 重新申请或同步数据。如果做了一次cpu与gpu之间的数据同步，会设置head状态为SYNCED。但mutable_cpu/gpu_data()会在之后将head
  // 设置为HEAD_AT_CPU/HEAD_AT_GPU
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void set_gpu_data(void* data);
  void* mutable_cpu_data();
  void* mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; }

  size_t size() { return size_; }

#ifndef CPU_ONLY
  // 将数据从CPU拷贝到GPU，使用异步拷贝。需要调用者之后synchronize流
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void to_cpu(); // 实际处理cpu和gpu数据同步的函数
  void to_gpu();
  void* cpu_ptr_; // 指向cpu/gpu内存块实际地址的指针
  void* gpu_ptr_;
  size_t size_; // 内存块字节大小
  SyncedHead head_;
  bool own_cpu_data_; // 判断当前cpu/gpu_ptr_变量所持有的内存块是否是自己分配的，如果是，则在需要释放内存块时要执行释放操作
  bool cpu_malloc_use_cuda_; // 判断cpu内存是否用了cuda pinned memory
  bool own_gpu_data_;
  int gpu_device_; // 保存分配gpu内存时的GPU ID

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
