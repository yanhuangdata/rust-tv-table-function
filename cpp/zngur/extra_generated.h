// this is only needed by macOS
#ifdef __APPLE__
#pragma once
template <> inline size_t __zngur_internal_size_of<uint8_t *>() {
  return sizeof(uint8_t *);
}

template <> inline uint8_t *__zngur_internal_data_ptr<size_t>(const size_t &t) {
  return const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(&t));
}

template <> inline void __zngur_internal_assume_init<size_t>(size_t &) {}
template <> inline void __zngur_internal_assume_deinit<size_t>(size_t &) {}

template <> inline size_t __zngur_internal_size_of<size_t>() {
  return sizeof(size_t);
}

template <> inline uint8_t *__zngur_internal_data_ptr<size_t *>(size_t *const &t) {
  return const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(&t));
}

template <> inline void __zngur_internal_assume_init<size_t *>(size_t *&) {}
template <> inline void __zngur_internal_assume_deinit<size_t *>(size_t *&) {}

template <> inline uint8_t *__zngur_internal_data_ptr<size_t const *>(size_t const *const &t) {
  return const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(&t));
}

template <> inline void __zngur_internal_assume_init<size_t const *>(size_t const *&) {}
template <> inline void __zngur_internal_assume_deinit<size_t const *>(size_t const *&) {}

template <> struct Ref<size_t> {
  Ref() {
    data = 0;
  }
  Ref(const size_t &t) {
    data = reinterpret_cast<size_t>(__zngur_internal_data_ptr(t));
  }

  size_t &operator*() {
    return *reinterpret_cast<size_t *>(data);
  }

private:
  size_t data;
  friend uint8_t * ::rust::__zngur_internal_data_ptr<Ref<size_t>>(const ::rust::Ref<size_t> &t);
};

template <> struct RefMut<size_t> {
  RefMut() {
    data = 0;
  }
  RefMut(size_t &t) {
    data = reinterpret_cast<size_t>(__zngur_internal_data_ptr(t));
  }

  size_t &operator*() {
    return *reinterpret_cast<size_t *>(data);
  }

private:
  size_t data;
  friend uint8_t * ::rust::__zngur_internal_data_ptr<RefMut<size_t>>(
      const ::rust::RefMut<size_t> &t);
};
#endif
