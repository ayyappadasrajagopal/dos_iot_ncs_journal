import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==============================
# 1. System model
A_c = np.array([
    [0, 1, 0, 0],
    [-13.94, -0.515, 0, 0.232],
    [0, 0, 0, 1],
    [1.13, 0.063, 0, -0.58]
])
B_c = np.array([
    [0, 0],
    [18.07, 0.15],
    [0, 0],
    [-0.29, 7.68]
])

dt = 0.002
A = np.eye(4) + A_c * dt
B = B_c * dt

# ==============================
# 2. LQR design
Q = np.diag([100, 1, 100, 1])
R = np.diag([0.05, 0.05])

def dlqr(A, B, Q, R, max_iter=1000, tol=1e-9):
    P = Q.copy()
    for _ in range(max_iter):
        P_next = A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A + Q
        if np.max(np.abs(P_next - P)) < tol:
            P = P_next
            break
        P = P_next
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    return K

K = dlqr(A, B, Q, R)

# ==============================
# 3. Integral control
Ki = np.array([2.0, 2.0])
z_int = np.zeros(2)

# ==============================
# 4. Desired reference
theta_d = np.deg2rad(10)
psi_d = np.deg2rad(15)
x_ref = np.array([theta_d, 0, psi_d, 0])

# ==============================
# 5. Simulation setup
sim_time = 5.0
num_steps = int(sim_time / dt)

time_history = []
theta_history = []
psi_history = []
u0_history = []
u1_history = []

# Initial state
theta = np.deg2rad(0)
psi = np.deg2rad(0)
theta_dot = 0
psi_dot = 0
x = np.array([theta, theta_dot, psi, psi_dot])

# ==============================
# 6. Update function for animation
def update(frame):
    global x, z_int

    t = frame * dt

    # Integral error
    err = np.array([x[0] - theta_d, x[2] - psi_d])
    z_int += err * dt

    # Control
    u = - K @ (x - x_ref) - Ki * z_int
    u = np.clip(u, -10, 10)

    # Simple simulation of dynamics (Euler integration)
    x = A @ x + B @ u

    # Store history
    time_history.append(t)
    theta_history.append(np.rad2deg(x[0]))
    psi_history.append(np.rad2deg(x[2]))
    u0_history.append(u[0])
    u1_history.append(u[1])

    # Update plots
    line_theta.set_data(time_history, theta_history)
    line_psi.set_data(time_history, psi_history)
    line_theta_ref.set_data(time_history, [np.rad2deg(theta_d)]*len(time_history))
    line_psi_ref.set_data(time_history, [np.rad2deg(psi_d)]*len(time_history))
    line_u0.set_data(time_history, u0_history)
    line_u1.set_data(time_history, u1_history)

    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    return line_theta, line_psi, line_theta_ref, line_psi_ref, line_u0, line_u1

# ==============================
# 7. Plot setup
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,8))

ax1.set_title("Pitch and Yaw Angles")
line_theta, = ax1.plot([], [], label="Pitch (deg)")
line_psi, = ax1.plot([], [], label="Yaw (deg)")
line_theta_ref, = ax1.plot([], [], 'r--', label="Pitch ref")
line_psi_ref, = ax1.plot([], [], 'g--', label="Yaw ref")
ax1.set_ylabel("Angle (deg)")
ax1.grid(True)
ax1.legend()

ax2.set_title("Control Inputs")
line_u0, = ax2.plot([], [], label="Pitch motor voltage")
line_u1, = ax2.plot([], [], label="Yaw motor voltage")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Voltage (V)")
ax2.grid(True)
ax2.legend()

ani = FuncAnimation(fig, update, frames=num_steps, interval=1, blit=False, repeat=False)
plt.show()
