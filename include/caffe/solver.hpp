#ifndef CAFFE_SOLVER_HPP_
#define CAFFE_SOLVER_HPP_
#include <boost/function.hpp>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/solver_factory.hpp"

namespace caffe {

/**
  * @brief Enumeration of actions that a client of the Solver may request by
  * implementing the Solver's action request function, which a
  * client may optionally provide in order to request early termination
  * or saving a snapshot without exiting. In the executable caffe, this
  * mechanism is used to allow the snapshot to be saved when stopping
  * execution with a SIGINT (Ctrl-C).
  */
  // 自定义的终端响应方式
  namespace SolverAction {
    enum Enum {
      NONE = 0,  // Take no special action.
      STOP = 1,  // Stop training. snapshot_after_train controls whether a
                 // snapshot is created.
      SNAPSHOT = 2  // Take a snapshot, and keep training.
    };
  }

/**
 * @brief Type of a function that returns a Solver Action enumeration.
 */
// 定义一个返回SolverAction的函数类型
typedef boost::function<SolverAction::Enum()> ActionCallback;

/**
 * @brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ApplyUpdate to compute a parameter update
 * given the current state of the Net parameters.
 */
// Solver的实现，主要包含Net和Net中参数更新的ApplyUpdate()函数
template <typename Dtype>
class Solver {
 public:
  // 初始化函数
  explicit Solver(const SolverParameter& param, const Solver* root_solver = NULL);
  explicit Solver(const string& param_file, const Solver* root_solver = NULL);
  void Init(const SolverParameter& param);
  void InitTrainNet();
  void InitTestNets();

  // Client of the Solver optionally may call this in order to set the function
  // that the solver uses to see what action it should take (e.g. snapshot or
  // exit training early).
  // 设置中断响应函数
  void SetActionFunction(ActionCallback func);
  // 返回中断所定义的操作
  SolverAction::Enum GetRequestedAction();
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  // Solver的主要函数，用于开始训练网络，是一个虚函数。内部会调用Step函数作实际的训练
  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
  // iters表示一共训练多少个iter
  void Step(int iters);
  // The Restore method simply dispatches to one of the
  // RestoreSolverStateFrom___ protected methods. You should implement these
  // methods to restore the state from the appropriate snapshot type.
  // 从snapshot文件中恢复状态
  void Restore(const char* resume_file);
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  // 将当前solver状态保存到snapshot文件
  void Snapshot();
  virtual ~Solver() {}
  // 返回当前Solver的状态
  inline const SolverParameter& param() const { return param_; }
  inline shared_ptr<Net<Dtype> > net() { return net_; }
  inline const vector<shared_ptr<Net<Dtype> > >& test_nets() {
    return test_nets_;
  }
  int iter() { return iter_; }

  // Invoked at specific points during an iteration
  // 定义回调类，内部保存一个回调对象的数组。在每个iter开始时会依次调用on_start(),在ForwardBackward完成后会依次调用on_gradients_ready()。
  class Callback {
   protected:
    virtual void on_start() = 0;
    virtual void on_gradients_ready() = 0;

    template <typename T>
    friend class Solver;
  };
  // 返回Solver的回调函数列表
  const vector<Callback*>& callbacks() const { return callbacks_; }
  void add_callback(Callback* value) {
    callbacks_.push_back(value);
  }

  // 检查是否有snapshot文件指定的目录存在且有写权限
  void CheckSnapshotWritePermissions();
  /**
   * @brief Returns the solver type.
   */
  // 返回Solver类型的字符串，在pb中定义了SolverType枚举类型：SGD,NESTEROV,ADAGRAD,RMSPROP,ADADEKTA,ADAM
  virtual inline const char* type() const { return ""; }

 protected:
  // Make and apply the update value for the current iteration.
  // 更新参数的函数，虚函数。在每个iter执行完ForwardBackward后调用
  virtual void ApplyUpdate() = 0;
  // 将当前状态保存到snapshot文件，可以是proto或HDF5格式
  string SnapshotFilename(const string extension);
  string SnapshotToBinaryProto();
  string SnapshotToHDF5();
  // The test routine
  // 用于test的函数
  void TestAll();
  void Test(const int test_net_id = 0);
  // 从snapshot文件中恢复的函数
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
  // 功能函数
  void DisplayOutputBlobs(const int net_id);
  void UpdateSmoothedLoss(Dtype loss, int start_iter, int average_loss);

  // 保存Solver的所有配置参数
  SolverParameter param_;
  // 保持当前训练的迭代次数，初始化时为0
  int iter_;
  int current_step_;
  // solver中的Net的指针
  shared_ptr<Net<Dtype> > net_;
  // 测试网络的指针数组
  vector<shared_ptr<Net<Dtype> > > test_nets_;
  // 训练前后回调函数对象的数组
  vector<Callback*> callbacks_;
  // 用于保存最近的SolverParameter.average_loss个loss的数组，这样display时给出的是最近average_loss的平均值smoothed_loss_
  vector<Dtype> losses_;
  Dtype smoothed_loss_;

  // The root solver that holds root nets (actually containing shared layers)
  // in data parallelism
  // 多线程执行是，保存指向root solver的指针，如果自己是root solver，这个指针是NULL
  const Solver* const root_solver_;

  // A function that can be set by a client of the Solver to provide indication
  // that it wants a snapshot saved and/or to exit early.
  // 定义对事件产生何种响应的函数
  ActionCallback action_request_function_;

  // True iff a request to stop early was received.
  // 判断是否要提前退出，程序有部分位置会检查这个变量
  bool requested_early_exit_;

  DISABLE_COPY_AND_ASSIGN(Solver);
};

/**
 * @brief Solver that only computes gradients, used as worker
 *        for multi-GPU training.
 */
// 用于多卡训练的Solver类
template <typename Dtype>
class WorkerSolver : public Solver<Dtype> {
 public:
  explicit WorkerSolver(const SolverParameter& param,
      const Solver<Dtype>* root_solver = NULL)
      : Solver<Dtype>(param, root_solver) {}

 protected:
  void ApplyUpdate() {}
  void SnapshotSolverState(const string& model_filename) {
    LOG(FATAL) << "Should not be called on worker solver.";
  }
  void RestoreSolverStateFromBinaryProto(const string& state_file) {
    LOG(FATAL) << "Should not be called on worker solver.";
  }
  void RestoreSolverStateFromHDF5(const string& state_file) {
    LOG(FATAL) << "Should not be called on worker solver.";
  }
};

}  // namespace caffe

#endif  // CAFFE_SOLVER_HPP_
