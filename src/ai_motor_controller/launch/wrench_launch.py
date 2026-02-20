# from launch import LaunchDescription
# from launch_ros.actions import Node
# import os

# def generate_launch_description():
#     # Path to the parameter YAML file
#     config_file = os.path.join(
#         os.path.dirname(__file__),
#         '..', 'config', 'pid_params.yaml'
#     )

#     return LaunchDescription([
#         Node(
#             package='ai_motor_controller',
#             executable='wrench',
#             name='wrench',
#             output='screen',
#             parameters=[config_file]
#         )
#     ])
