#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

#include <boost/lockfree/queue.hpp>

#include "Common.hpp"
#include "ThreadPoolTask.hpp"

#define USE_YIELD

class ThreadPool {
public:
  // Get the global singleton instance.
  static ThreadPool &instance(void);

  ThreadPool();
  ThreadPool(size_t concurrency);
  ThreadPool(size_t concurrency, size_t queueSize);

  ~ThreadPool();

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool(ThreadPool &&) = delete;

  ThreadPool &operator=(const ThreadPool &) = delete;
  ThreadPool &operator=(ThreadPool &&) = delete;

  //  enqueue_task
  //
  //  Runs the given function on one of the thread pool
  //  threads in First In First Out (FIFO) order
  //
  //  Arguments
  //    task - Function or functor to be called on the
  //           thread pool, takes an arbitary number of
  //           arguments and arbitary return type.
  //    args - Arguments for task, cannot be std::move'ed
  //           if such parameters must be used, use a
  //           lambda and capture via move then move
  //           the lambda.
  //
  //  Result
  //    Signals when the task has completed with either
  //    success or an exception. Also results in an
  //    exception if the thread pool is destroyed before
  //    execution has begun.
  template <typename Func, typename... Args>
  auto Execute(Func &&task, Args &&... args)
      -> std::future<decltype(task(std::forward<Args>(args)...))> {
    //  Return type of the functor, can be void via
    //  specilisation of task_package_impl
    using R = decltype(task(std::forward<Args>(args)...));

    auto promise = std::promise<R>{};
    auto future = promise.get_future();
    auto bound_task = std::bind(std::forward<Func>(task), std::forward<Args>(args)...);

    // ensures no memory leak if push throws (it shouldn't but to be safe)
    auto package_ptr = std::make_unique<task_package_impl<R, decltype(bound_task)>>(
        std::move(bound_task), std::move(promise));

    tasks.push(static_cast<task_package *>(package_ptr.get()));

    // no longer in danger, can revoke ownership so
    // tasks is not left with dangling reference
    package_ptr.release();

#ifndef USE_YIELD
    wakeup_signal.notify_one();
#endif

    return future;
  };

  unsigned NumThreads(void) const { return threads.size(); }

private:
  std::vector<std::thread> threads;
  std::atomic<bool> shutdown_flag;
  boost::lockfree::queue<task_package *> tasks;

#ifndef USE_YIELD
  std::condition_variable wakeup_signal;
  std::mutex wakeup_mutex;
#endif

  bool pop_task(std::unique_ptr<task_package> &out);
};
