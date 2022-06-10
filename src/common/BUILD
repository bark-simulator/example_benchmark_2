
py_library(
  name = "pyhelpers",
  srcs = ["pyhelpers.py"],
)

py_library(
  name = "debug_settings",
  srcs = ["debug_settings.py"],
)

py_library(
  name = "plotting",
  srcs = ["plotting.py"],
  data=["//doc/common:latex"]
)

py_library(
  name = "export",
  srcs = ["export_table.py",
          "export_image.py"],
)


py_library(
  name = "common",
  deps = [
      ":pyhelpers",
      ":plotting",
      ":export",
      ":debug_settings"
  ],
  visibility = ["//visibility:public"],
)


py_test(
  name = "test_ray",
  srcs = ["test_ray.py"]
)