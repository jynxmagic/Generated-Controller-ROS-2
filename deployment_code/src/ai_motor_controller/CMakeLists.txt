cmake_minimum_required(VERSION 3.5)
project(ai_motor_controller)

# Default to C99
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_ros_com REQUIRED)
# further dependencies manually.
# find_package(<dependency> REQUIRED)


include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(lemniscate src/lemniscate.cpp)
add_executable(lissajous src/lissajous.cpp)


#add_executable(wrench src/wrench_disturbance.cpp)

ament_target_dependencies(lemniscate
  rclcpp
  px4_msgs
  px4_ros_com
  Eigen3
)

ament_target_dependencies(lissajous
  rclcpp
  px4_msgs
  px4_ros_com
  Eigen3
)


#ament_target_dependencies(wrench
#  rclcpp
#  px4_msgs
#  px4_ros_com
#  Eigen3
#)

install(TARGETS
  lemniscate
  lissajous
#  wrench
  DESTINATION lib/${PROJECT_NAME})



# Export information to downstream packages
ament_export_dependencies(ament_cmake rclcpp rosidl_default_runtime eigen3_cmake_module Eigen3 px4_msgs px4_ros_com)

# Install launch files.
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME}/)



if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
