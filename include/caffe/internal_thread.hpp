#ifndef CAFFE_INTERNAL_THREAD_HPP_
#define CAFFE_INTERNAL_THREAD_HPP_

#include "caffe/common.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class thread; }

namespace caffe {

/**
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virtual function InternalThreadEntry.
 */
// 一般作为基类，给子类提供一个附加线程，附加线程执行InternalThreadEntry()
// 中的操作
class InternalThread {
 public:
  InternalThread() : thread_() {}
  virtual ~InternalThread();

  /**
   * Caffe's thread local state will be initialized using the current
   * thread values, e.g. device id, solver index etc. The random seed
   * is initialized using caffe_rng_rand.
   */
  // 线程的启动,会用entry()函数去初始化线程，entry()函数设置环境变量，然后
  // 执行InternalThreadEntry()函数
  void StartInternalThread();

  /** Will not return until the internal thread has exited. */
  // 阻塞等待线程结束
  void StopInternalThread();

  // 检查线程是否在运行
  bool is_started() const;

 protected:
  /* Implement this method in your subclass
      with the code you want your thread to run. */
  // 在子类中定义，附加的线程会执行这个函数，一般是一个while(!must_stop()){}循环
  virtual void InternalThreadEntry() {}

  /* Should be tested when running loops to exit when requested. */
  // 在子类InternalThreadEntry()函数中用于退出循环
  bool must_stop();

 private:
  // 初始化thread_线程，设置线程中Caffe的状态，然后调用InternalThreadEntry()
  void entry(int device, Caffe::Brew mode, int rand_seed, int solver_count,
      bool root_solver);

  shared_ptr<boost::thread> thread_;
};

}  // namespace caffe

#endif  // CAFFE_INTERNAL_THREAD_HPP_
