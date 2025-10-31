import zmq
import threading
import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

# =========================
# ====== ZMQ CONFIG =======
# =========================
MY_IP   = "10.128.14.249"
MY_PORT = 5557

PEER_IP   = "10.128.7.67"
PEER_PORT = 5555

context = zmq.Context()

# PUB: for control outputs u0,u1
pub_socket = context.socket(zmq.PUB)
pub_socket.bind(f"tcp://{MY_IP}:{MY_PORT}")

# SUB: for sensor inputs theta,psi,(theta_dot,psi_dot optional)
sub_socket = context.socket(zmq.SUB)
sub_socket.connect(f"tcp://{PEER_IP}:{PEER_PORT}")
sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # subscribe to all

# =========================
# ====== CONTROLLER =======
# =========================
# Continuous-time model (given)
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

# Discretization step for control loop (nominal; we’ll integrate with actual dt_meas)
DT_NOM = 0.01  # 100 Hz control (adjust if your sensor publish rate differs)
A = np.eye(4) + A_c * DT_NOM
B = B_c * DT_NOM

# LQR weights (given)
Q = np.diag([100, 1, 100, 1])
R = np.diag([0.05, 0.05])

def dlqr(A, B, Q, R, max_iter=1000, tol=1e-9):
    P = Q.copy()
    for _ in range(max_iter):
        BT_P = B.T @ P
        S = R + BT_P @ B
        K_tmp = np.linalg.inv(S) @ BT_P @ A
        P_next = A.T @ (P - P @ B @ np.linalg.inv(S) @ BT_P) @ A + Q
        if np.max(np.abs(P_next - P)) < tol:
            P = P_next
            break
        P = P_next
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    return K

K = dlqr(A, B, Q, R)

# Integral action
Ki = np.array([2.0, 2.0])   # integral gains for [theta, psi]
z_int = np.zeros(2)

# References (deg) -> will be converted to rad
THETA_REF_DEG = 10.0
PSI_REF_DEG   = 15.0
theta_d = np.deg2rad(THETA_REF_DEG)
psi_d   = np.deg2rad(PSI_REF_DEG)
x_ref   = np.array([theta_d, 0.0, psi_d, 0.0])

# Output clip (e.g., motor voltages)
U_MAX = 10.0

# State, with derivative estimation if not provided
x = np.zeros(4)  # [theta, theta_dot, psi, psi_dot]
_last_theta = None
_last_psi   = None
_last_t     = None

# Simple LPF for derivative estimates
DERIV_ALPHA = 0.2  # 0..1 (lower = more smoothing)
theta_dot_est = 0.0
psi_dot_est   = 0.0

def parse_sensor_message(msg_str):
    """
    Accepts:
      - "theta,psi" (angles only)
      - "theta,theta_dot,psi,psi_dot"
    Auto-detects degrees vs radians (if any |val| > pi => degrees).
    Returns: theta, theta_dot, psi, psi_dot (all in radians / radians-per-second).
    If derivatives absent, returns None for dot terms (caller estimates).
    """
    parts = [float(x.strip()) for x in msg_str.strip().split(",") if x.strip() != ""]
    if len(parts) not in (2, 4):
        raise ValueError("Expect 2 or 4 comma-separated values.")

    # Unit auto-detection on angle fields
    angle_vals = [parts[0], parts[2]] if len(parts) == 4 else parts[:2]
    in_degrees = any(abs(v) > np.pi for v in angle_vals)

    if len(parts) == 2:
        theta, psi = parts[0], parts[1]
        if in_degrees:
            theta = np.deg2rad(theta)
            psi   = np.deg2rad(psi)
        return theta, None, psi, None

    else:
        theta, theta_dot, psi, psi_dot = parts
        if in_degrees:
            theta     = np.deg2rad(theta)
            psi       = np.deg2rad(psi)
            # If dots came in deg/s, convert to rad/s as well (assume same units as angles)
            theta_dot = np.deg2rad(theta_dot)
            psi_dot   = np.deg2rad(psi_dot)
        return theta, theta_dot, psi, psi_dot

def control_step_from_sensor(theta, theta_dot_in, psi, psi_dot_in, t_now):
    """
    Build state x (with derivative estimation if needed), update integral,
    compute control u, clip, and return u, state, and display angles in deg.
    """
    global x, z_int, _last_theta, _last_psi, _last_t, theta_dot_est, psi_dot_est

    # Derivative estimation if needed
    if _last_t is None:
        _last_t = t_now
        _last_theta = theta
        _last_psi = psi

    dt_meas = max(1e-4, t_now - _last_t)

    if theta_dot_in is None:
        raw = (theta - _last_theta) / dt_meas
        theta_dot_est = (1 - DERIV_ALPHA) * theta_dot_est + DERIV_ALPHA * raw
        theta_dot = theta_dot_est
    else:
        theta_dot = theta_dot_in

    if psi_dot_in is None:
        raw = (psi - _last_psi) / dt_meas
        psi_dot_est = (1 - DERIV_ALPHA) * psi_dot_est + DERIV_ALPHA * raw
        psi_dot = psi_dot_est
    else:
        psi_dot = psi_dot_in

    _last_t = t_now
    _last_theta = theta
    _last_psi = psi

    # State vector
    x = np.array([theta, theta_dot, psi, psi_dot], dtype=float)

    # Integral error on angles
    err = np.array([x[0] - theta_d, x[2] - psi_d])
    z_int = z_int + err * dt_meas  # accumulate with measured dt

    # LQR + integral control
    u = - K @ (x - x_ref) - Ki * z_int
    u = np.clip(u, -U_MAX, U_MAX)

    # For GUI display (received angles in deg)
    theta_deg = np.rad2deg(theta)
    psi_deg   = np.rad2deg(psi)
    return u, theta_deg, psi_deg

# =========================
# ========  GUI  ==========
# =========================
root = tk.Tk()
root.title("IoT Server Platform Dashboard")
root.geometry("1250x700")
root.configure(bg="#101820")

title_label = tk.Label(
    root, text="IoT SERVER PLATFORM",
    font=("Helvetica", 22, "bold"), bg="#101820", fg="#00ADEF"
)
title_label.pack(pady=10)

main_frame = tk.Frame(root, bg="#101820")
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Left log panel
log_frame = tk.Frame(main_frame, bg="#101820")
log_frame.pack(side="left", fill="y", padx=10)

tk.Label(
    log_frame, text="Message Log:", font=("Helvetica", 14, "bold"),
    bg="#101820", fg="white"
).pack(anchor="w")

msg_display = tk.Text(
    log_frame, height=18, width=45, bg="#1B2735", fg="white", font=("Consolas", 11)
)
msg_display.pack(pady=5)

tk.Label(
    log_frame, text="Transmitted Data Log:",
    font=("Helvetica", 14, "bold"), bg="#101820", fg="white"
).pack(anchor="w", pady=(20, 0))

data_display = tk.Text(
    log_frame, height=15, width=45, bg="#1B2735", fg="#00FF7F", font=("Consolas", 11)
)
data_display.pack(pady=5)

def log_message(text):
    msg_display.insert(tk.END, text + "\n")
    msg_display.see(tk.END)

def log_data(text):
    data_display.insert(tk.END, text + "\n")
    data_display.see(tk.END)

# Right plots (Transmitted vs Received)
fig = Figure(figsize=(9, 6), dpi=100)
fig.patch.set_facecolor("#101820")
ax_tx = fig.add_subplot(211)
ax_rx = fig.add_subplot(212)
for ax in (ax_tx, ax_rx):
    ax.set_facecolor("#1B2735")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
ax_tx.set_title("Transmitted Control (u0, u1) [V]", color="white")
ax_rx.set_title("Received Angles (θ, ψ) [deg]", color="white")
ax_tx.set_ylabel("Value")
ax_rx.set_ylabel("Value")
ax_rx.set_xlabel("Samples")

canvas = FigureCanvasTkAgg(fig, master=main_frame)
canvas.get_tk_widget().pack(side="right", fill="both", expand=True, padx=10)

# Data buffers
x_data_tx, y1_tx, y2_tx = [], [], []  # u0,u1
x_data_rx, y1_rx, y2_rx = [], [], []  # theta_deg, psi_deg

line_tx1, = ax_tx.plot([], [], color="#00ADEF", label="u0", linewidth=2)
line_tx2, = ax_tx.plot([], [], color="#FF6F61", label="u1", linewidth=2)
line_rx1, = ax_rx.plot([], [], color="#00FF7F", label="θ (deg)", linewidth=2)
line_rx2, = ax_rx.plot([], [], color="#FFBF00", label="ψ (deg)", linewidth=2)

ax_tx.legend(facecolor="#1B2735", labelcolor="white")
ax_rx.legend(facecolor="#1B2735", labelcolor="white")
canvas.draw()

max_y_tx = min_y_tx = None
max_y_rx = min_y_rx = None

def _rescale_axis(ax, ys, cur_min, cur_max):
    if not ys:
        return cur_min, cur_max
    cmax = max(ys)
    cmin = min(ys)
    cur_max = cmax if cur_max is None else max(cur_max, cmax)
    cur_min = cmin if cur_min is None else min(cur_min, cmin)
    margin = (cur_max - cur_min) * 0.1 if cur_max != cur_min else 1.0
    ax.set_ylim(cur_min - margin, cur_max + margin)
    return cur_min, cur_max

def update_tx_graph(u0, u1):
    global max_y_tx, min_y_tx
    x_data_tx.append(len(x_data_tx))
    y1_tx.append(u0)
    y2_tx.append(u1)
    if len(x_data_tx) > 300:
        x_data_tx.pop(0); y1_tx.pop(0); y2_tx.pop(0)
    line_tx1.set_data(range(len(x_data_tx)), y1_tx)
    line_tx2.set_data(range(len(x_data_tx)), y2_tx)
    min_y_tx, max_y_tx = _rescale_axis(ax_tx, y1_tx + y2_tx, min_y_tx, max_y_tx)
    ax_tx.set_xlim(0, len(x_data_tx))
    canvas.draw_idle()

def update_rx_graph(theta_deg, psi_deg):
    global max_y_rx, min_y_rx
    x_data_rx.append(len(x_data_rx))
    y1_rx.append(theta_deg)
    y2_rx.append(psi_deg)
    if len(x_data_rx) > 300:
        x_data_rx.pop(0); y1_rx.pop(0); y2_rx.pop(0)
    line_rx1.set_data(range(len(x_data_rx)), y1_rx)
    line_rx2.set_data(range(len(x_data_rx)), y2_rx)
    min_y_rx, max_y_rx = _rescale_axis(ax_rx, y1_rx + y2_rx, min_y_rx, max_y_rx)
    ax_rx.set_xlim(0, len(x_data_rx))
    canvas.draw_idle()

# =========================
# ====== RX THREAD ========
# =========================
def receive_and_control():
    while True:
        try:
            msg = sub_socket.recv_string()
        except Exception as e:
            log_message(f"[RECV ERROR] {e}")
            continue

        t_now = time.monotonic()
        try:
            theta, theta_dot_in, psi, psi_dot_in = parse_sensor_message(msg)
        except Exception as e:
            root.after(0, log_message, f"[RECEIVED] {msg} (parse error: {e})")
            continue

        # Compute control
        u, theta_deg, psi_deg = control_step_from_sensor(theta, theta_dot_in, psi, psi_dot_in, t_now)
        u0, u1 = float(u[0]), float(u[1])

        # Publish control (u0,u1)
        out_msg = f"{u0:.4f},{u1:.4f}"
        try:
            pub_socket.send_string(out_msg)
        except Exception as e:
            root.after(0, log_message, f"[SEND ERROR] {e}")

        # Update GUI safely
        root.after(0, log_message, f"[RECEIVED] {msg}")
        root.after(0, update_rx_graph, theta_deg, psi_deg)
        root.after(0, log_message, f"[SENT CTRL] {out_msg}")
        root.after(0, log_data,   f"{u0:7.3f}, {u1:7.3f}")
        root.after(0, update_tx_graph, u0, u1)

threading.Thread(target=receive_and_control, daemon=True).start()

# =========================
# ====== MAIN LOOP ========
# =========================
root.mainloop()
