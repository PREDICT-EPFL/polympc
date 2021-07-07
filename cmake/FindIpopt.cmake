find_package(PkgConfig QUIET)

# Check to see if pkgconfig is installed.
pkg_check_modules(PC_Ipopt ipopt)

# Definitions
set(Ipopt_DEFINITIONS ${PC_Ipopt_CFLAGS_OTHER})

# Include directories
find_path(Ipopt_INCLUDE_DIRS
    NAMES IpIpoptNLP.hpp
    HINTS ${PC_Ipopt_INCLUDEDIR}
    PATHS "${CMAKE_INSTALL_PREFIX}/include")

# Libraries
find_library(Ipopt_LIBRARIES
    NAMES ipopt
    HINTS ${PC_Ipopt_LIBDIR})

# Version
set(Ipopt_VERSION ${PC_IPOPT_VERSION})

# Set (NAME)_FOUND if all the variables and the version are satisfied.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Ipopt
    FAIL_MESSAGE  DEFAULT_MSG
    REQUIRED_VARS Ipopt_INCLUDE_DIRS Ipopt_LIBRARIES
    VERSION_VAR   Ipopt_VERSION)
