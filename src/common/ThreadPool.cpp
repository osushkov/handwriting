#include "ThreadPool.hpp"

#include <exception>
#include <utility>
#include <iostream>


template<typename T>
constexpr T zero(T) {
  return 0;
}

static const unsigned NUM_THREADS = 4;
static ThreadPool singletonInstance(NUM_THREADS);

ThreadPool& ThreadPool::instance(void) {
  return singletonInstance;
}

ThreadPool::ThreadPool() : ThreadPool(std::thread::hardware_concurrency()) {}
ThreadPool::ThreadPool(size_t concurrency) : ThreadPool(concurrency, 128) {}

ThreadPool::ThreadPool(size_t concurrency, size_t queueSize)
    :  threads(), shutdown_flag(false), tasks(queueSize)
#ifndef USE_YIELD
    , wakeup_signal(), wakeup_mutex()
#endif
{
  // This is more efficient than creating the 'threads' vector with
  // size constructor and populating with std::generate since
  // std::thread objects will be constructed only to be replaced
  threads.reserve(concurrency);

  for (auto a = zero(concurrency); a < concurrency; ++a) {
    // emplace_back so thread is constructed in place
    threads.emplace_back([this]() {
      // checks whether parent ThreadPool is being destroyed,
      // if it is, stop running.
      while (!shutdown_flag.load(std::memory_order_relaxed)) {
        auto current_task_package = std::unique_ptr<task_package>{nullptr};

        // use pop_task so we only ever have one reference to the
        // task_package
        if (pop_task(current_task_package)) {
          current_task_package->run_task();
        } else {
          // rather than spinning, give up thread time to other things
#ifdef USE_YIELD
          std::this_thread::yield();
#else
          auto lock = std::unique_lock<std::mutex>(wakeup_mutex);
          wakeup_signal.wait(lock, [this](){ return !tasks.empty() || shutdown_flag; });
#endif
        }
      }
  });
  }
}

ThreadPool::~ThreadPool() {
  // signal that threads should not perform any new work
  shutdown_flag.store(true);

#ifndef USE_YIELD
  wakeup_signal.notify_all();
#endif

  // wait for work to complete then destroy thread
  for (auto && thread : threads) {
    thread.join();
  }

  auto current_task_package = std::unique_ptr<task_package>{nullptr};

  // signal to each uncomplete task that it will not complete due to
  // ThreadPool destruction
  while (pop_task(current_task_package)) {
    auto except = std::runtime_error("Could not perform task before ThreadPool destruction");
    current_task_package->set_exception(std::make_exception_ptr(except));
  }
}

bool ThreadPool::pop_task(std::unique_ptr<task_package> & out) {
  task_package *temp_ptr = nullptr;

  if (tasks.pop(temp_ptr)) {
    out.reset(temp_ptr);
    return true;
  }
  return false;
}
