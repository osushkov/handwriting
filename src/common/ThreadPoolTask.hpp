#pragma once

#include <future>

struct task_package {
  virtual ~task_package() {};

  void run_task() noexcept {
    try {
        run();
    } catch (...) {
        set_exception(std::current_exception());
    }
  }

  virtual void run() = 0;
  virtual void set_exception(std::exception_ptr except_ptr) = 0;
};

template<typename R, typename Func>
struct task_package_impl : public task_package {
  task_package_impl(Func&& func, std::promise<R>&& promise)
    : promise(std::forward<std::promise<R>>(promise)), func(std::forward<Func>(func)) {};

  virtual void run() {
    promise.set_value(func());
  }

  virtual void set_exception(std::exception_ptr except_ptr) {
    promise.set_exception(except_ptr);
  }

  std::promise<R> promise;
  Func func;
};

template<typename Func>
struct task_package_impl<void, Func> : public task_package {
  task_package_impl(Func&& func, std::promise<void>&& promise)
    : promise(std::forward<std::promise<void>>(promise)), func(std::forward<Func>(func)) {};

  virtual void run() {
      func();
      promise.set_value();
  }

  virtual void set_exception(std::exception_ptr except_ptr) {
      promise.set_exception(except_ptr);
  }

  std::promise<void> promise;
  Func func;
};
