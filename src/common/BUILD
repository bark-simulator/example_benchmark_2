
py_library(
  name = "pyhelpers",
  srcs = ["pyhelpers.py"],
)

py_library(
  name = "debug_settings",
  srcs = ["debug_settings.py"],
)



py_library(
  name = "common",
  deps = [
      ":pyhelpers",
      ":debug_settings"
  ],
  visibility = ["//visibility:public"],
)
