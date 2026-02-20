from scipy.spatial import cKDTree as KDTree
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import glob
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def generate_waypoints():
    half_points = 32
    total_points = half_points * 2  # 64 total waypoints
    start_x = 0.0
    start_y = 0.0

    # Altitude parameters
    base_z = 2.5  # Corresponds to t_z_ in the original code
    amplitude_z = 0.5

    max_radius = 1.5
    angle_increment = 2.0 * np.pi / half_points

    # Initialize array to hold the waypoints
    waypoints = np.zeros((total_points, 3))

    # --- First 32 waypoints: normal z oscillation ---
    # z = base_z + sin(angle)
    for i in range(half_points):
        angle = angle_increment * i
        x = start_x + max_radius * np.sin(angle)
        y = start_y + max_radius * np.sin(2 * angle)
        z = base_z + amplitude_z * np.sin(angle)
        waypoints[i, :] = [x, y, z]

    # --- Second 32 waypoints: reversed z oscillation ---
    # z = base_z - sin(angle)
    for i in range(half_points):
        angle = angle_increment * i
        index = half_points + i
        # Duplicate x and y from the first half
        x, y = waypoints[i, 0], waypoints[i, 1]
        z = base_z - amplitude_z * np.sin(angle)
        waypoints[index, :] = [x, y, z]

    return waypoints


true_waypoints = generate_waypoints()  # 64 x 3 array

# Create a normalized progress vector for the true setpoints (0 to 1)
true_progress = np.linspace(0, 1, 64)


goal_points = true_waypoints
goal_tree = KDTree(goal_points)


# def plot_traj(file_loc, name=""):
#     df = pd.read_csv(file_loc)
#     df = df[df.z > 0]
#     #plot xyz in 3d plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot(df.x, df.y, df.z, color="g", markersize=0.1)
#     ax.plot(waypoints[:, 1], waypoints[:, 0], waypoints[:, 2], color="b", markersize=0.1)
#     ax.set_zlim(0, 5)
#     ax.set_title(name)

#     actual_points = df[['x', 'y', 'z']].values
#     distances, indices = goal_tree.query(actual_points)

#     print(f"{name}, MSE of the trajectory: ", np.mean(distances ** 2))
#     print(f"{name}, Max distance from the trajectory: ", np.max(distances))
#     print(f"{name}, RMSE of trajecotery: ", np.sqrt(np.mean(distances ** 2)))

#     # label df.x as "quadcopter path" and df.gx as "ground truth"
#     ax.legend(['Quadcopter path', 'Desired path', 'Start pos'])
#     #  label x, y, z axis as "x", "y", "z"
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.show()

# plot_traj("./pid_motor_fault.csv", "pid faulty motor")
# #plot_traj("./pid_noisy.csv", "pid noisy obs") # time_taken = 692.181s
# plot_traj("./no_error_pid_and_rl.csv", "PID and RL - Nominal conditions") # time_taken = 289.061s
# plot_traj("./no_error_pid.csv", "PID - Nominal conditions") # time_taken = 308.846s


def plot_data(file_path, title):
    csv_files = glob.glob(file_path)

    lap_distances = []

    lap_times = []

    # plot each individual lap
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot(true_waypoints[:, 0], true_waypoints[:, 1],
    #         true_waypoints[:, 2], color="b", markersize=0.1)

    for file in csv_files:
        df = pd.read_csv(file)
        mse = (df.x - df.gx) ** 2 + (df.y - df.gy) ** 2
        mse = np.mean(mse)
        print("mse: ", mse)
        # Assume columns: time, x, y, z, type, lap; we ignore time and type.
        ax.plot(df.x, df.y, df.z, color="g", markersize=0.1)
        ax.plot(df.gx, df.gy, df.gz, color="b", markersize=0.1)
        print("man and max z, ", df.z.min(), df.z.max())
        # # Query the goal_tree: for every measured point, find its distance to the true path.
        # distances, _ = goal_tree.query(measured_points)
        # lap_distances.append(distances)
        # lap_time = group['time'].iloc[-1] - group['time'].iloc[0]
        # lap_times.append(lap_time)
        ax.set_zlim(0, 1)
    plt.title(title)
    plt.show()

    # print(f"MSE {title}: ", np.mean(np.array(lap_distances) ** 2))
    # print(f"Max distance from the trajectory {title}: ", np.max(np.array(lap_distances)))
    # print(f"RMSE {title}: ", np.sqrt(np.mean(np.array(lap_distances) ** 2)))
    # Additionally, print overall min, max, and average lap time.
    lap_times = np.array(lap_times)
    print("\nLap Times:")
    print("  Minimum:", lap_times.min())
    print("  Average:", lap_times.mean())
    print("  Maximum:", lap_times.max())
    resampled_errors = []
    for dist in lap_distances:
        old_idx = np.linspace(0, 1, len(dist))
        new_idx = np.linspace(0, 1, 64)
        resampled = np.interp(new_idx, old_idx, dist)
        resampled_errors.append(resampled)
    resampled_errors = np.array(resampled_errors)  # shape (num_laps, 64)

    # Compute the mean error at each waypoint (across all laps)
    error_envelope = resampled_errors.mean(axis=0)  # shape (64,)

    up = np.array([0, 0, 1])
    n_points = true_waypoints.shape[0]
    tangents = np.zeros_like(true_waypoints)

    # Compute tangent vectors (using forward/backward differences)
    for i in range(n_points):
        if i == 0:
            tangent = true_waypoints[1] - true_waypoints[0]
        elif i == n_points - 1:
            tangent = true_waypoints[-1] - true_waypoints[-2]
        else:
            tangent = (true_waypoints[i+1] - true_waypoints[i-1]) * 0.5
        norm = np.linalg.norm(tangent)
        if norm == 0:
            tangent = np.array([1, 0, 0])
        else:
            tangent = tangent / norm
        tangents[i] = tangent

    right_vectors = np.zeros_like(true_waypoints)
    for i in range(n_points):
        # Cross product: right = tangent x up
        right = np.cross(tangents[i], up)
        norm = np.linalg.norm(right)
        if norm < 1e-6:
            # If tangent is nearly vertical, default to x-axis.
            right = np.array([1, 0, 0])
        else:
            right = right / norm
        right_vectors[i] = right

    P_plus = true_waypoints + (error_envelope[:, None] * right_vectors)
    P_minus = true_waypoints - (error_envelope[:, None] * right_vectors)

    # ----------------------------
    # 5. Plot the true path in blue and the error tube (orange area) around it
    # ----------------------------
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the true path (blue)
    ax.plot(true_waypoints[:, 0], true_waypoints[:, 1], true_waypoints[:, 2],
            color='blue', linewidth=2, label='True Path')

    # Plot the tube edges (orange curves)
    ax.plot(P_plus[:, 0], P_plus[:, 1], P_plus[:, 2],
            color='orange', linewidth=2, label='Error Envelope')
    ax.plot(P_minus[:, 0], P_minus[:, 1], P_minus[:, 2],
            color='orange', linewidth=2)

    polys = []
    for i in range(n_points - 1):
        quad = [P_plus[i], P_plus[i+1], P_minus[i+1], P_minus[i]]
        polys.append(quad)

    tube = Poly3DCollection(polys, facecolor='orange', alpha=0.3)
    ax.add_collection3d(tube)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title} - Expected vs Actual Trajectory")
    ax.set_zlim(0, 5)
    ax.legend()
    plt.tight_layout()
    plt.show()


# plot_data("./faultless_fo8_adhoc_rvolmea.csv", "RVOLMEA - Training trajectory")
plot_data("traj.csv", "Generated controller - SITL")

# plot_data("./simulated_data/pid_faulty_motor/*.csv", "PID + DOB - Motor Fault")
# plot_data("./simulated_data/pid_nominal/*.csv", "PID + DOB - Nominal conditions")
# plot_data("./simulated_data/rl_nominal/*.csv", "PID + RL - Nominal conditions")

# grouped.columns = ['_'.join(col) for col in grouped.columns]
# grouped.reset_index(inplace=True)

# mean_x = grouped['x_mean'].to_numpy()
# mean_y = grouped['y_mean'].to_numpy()
# mean_z = grouped['z_mean'].to_numpy()
# std_x = grouped['x_std'].to_numpy()
# std_y = grouped['y_std'].to_numpy()
# std_z = grouped['z_std'].to_numpy()

# # Determine bin centers (which we will use to map the true trajectory)
# bin_centers = (bins[:-1] + bins[1:]) / 2

# # Interpolate the true trajectory to the bin centers
# interp_true_x = interp1d(true_progress, true_waypoints[:, 0], kind='linear')
# interp_true_y = interp1d(true_progress, true_waypoints[:, 1], kind='linear')
# interp_true_z = interp1d(true_progress, true_waypoints[:, 2], kind='linear')


# true_x = interp_true_x(bin_centers)
# true_y = interp_true_y(bin_centers)
# true_z = interp_true_z(bin_centers)


# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot the true trajectory in blue
# ax.plot(true_x, true_y, true_z, color='blue', label='True Trajectory', linewidth=2)

# # Plot the average measured path in orange
# ax.plot(mean_x, mean_y, mean_z, color='orange', label='Average Path', linewidth=2)


# for i in range(len(mean_x)):
#     # X error bar
#     ax.plot([mean_x[i] - std_x[i], mean_x[i] + std_x[i]],
#             [mean_y[i], mean_y[i]],
#             [mean_z[i], mean_z[i]], color='orange')
#     # Y error bar
#     ax.plot([mean_x[i], mean_x[i]],
#             [mean_y[i] - std_y[i], mean_y[i] + std_y[i]],
#             [mean_z[i], mean_z[i]], color='orange')
#     # Z error bar
#     ax.plot([mean_x[i], mean_x[i]],
#             [mean_y[i], mean_y[i]],
#             [mean_z[i] - std_z[i], mean_z[i] + std_z[i]], color='orange')


# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.legend()
# ax.set_title("True Trajectory (Blue) vs. Average Path (Orange) with Error Bars")


# plt.tight_layout()
# plt.show()
