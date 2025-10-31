import zmq
import threading
import time
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter  # for 2-decimal axis ticks
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
], dtype=float)
B_c = np.array([
    [0, 0],
    [18.07, 0.15],
    [0, 0],
    [-0.29, 7.68]
], dtype=float)

# Nominal discretization step (used for LQR design only)
DT_NOM = 0.01  # 100 Hz design
A = np.eye(4) + A_c * DT_NOM
B = B_c * DT_NOM

# LQR weights (given)
Q = np.diag([100, 1, 100, 1]).astype(float)
R = np.diag([0.05, 0.05]).astype(float)

def dlqr(A, B, Q, R, max_iter=1000, tol=1e-9):
    P = Q.copy()
    for _ in range(max_iter):
        BT_P = B.T @ P
        S = R + BT_P @ B
        P_next = A.T @ (P - P @ B @ np.linalg.inv(S) @ BT_P) @ A + Q
        if np.max(np.abs(P_next - P)) < tol:
            P = P_next
            break
        P = P_next
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    return K

K = dlqr(A, B, Q, R)

# Integral action (with anti-windup)
Ki = np.array([2.0, 2.0])   # integral gains for [theta, psi]
z_int = np.zeros(2)

# ---- Setpoints (GUI-controlled; keep radians internally) ----
THETA_REF_DEG_INIT = 10.0
PSI_REF_DEG_INIT   = 15.0
theta_ref_cmd = np.deg2rad(THETA_REF_DEG_INIT)  # target from GUI (rad)
psi_ref_cmd   = np.deg2rad(PSI_REF_DEG_INIT)    # target from GUI (rad)

# The *applied* references ramp toward the command at a limited rate:
theta_d = theta_ref_cmd
psi_d   = psi_ref_cmd

def ramp_to(current, target, rate, dt):
    """Slew-rate limit in rad/s."""
    if rate <= 0 or dt <= 0:
        return target
    step = rate * dt
    if target > current:
        return min(target, current + step)
    else:
        return max(target, current - step)

# Output clip (e.g., motor voltages)
U_MAX = 10.0
u_last = np.zeros(2)  # for prediction

# State, with derivative estimation if not provided
x = np.zeros(4)  # [theta, theta_dot, psi, psi_dot]
_last_theta = None
_last_psi   = None
_last_t     = None

# Sensor smoothing (exponential)
theta_filt = 0.0
psi_filt   = 0.0
theta_dot_est = 0.0
psi_dot_est   = 0.0

def parse_sensor_message(msg_str):
    """
    Accepts:
      - "theta,psi" (angles only)
      - "theta,theta_dot,psi,psi_dot"
    Auto-detects degrees vs radians (if any |val| > pi => degrees).
    Returns: theta, theta_dot, psi, psi_dot (radians / rad/s).
    If derivatives absent, returns None for dot terms (caller estimates).
    """
    parts = [float(x.strip()) for x in msg_str.strip().split(",") if x.strip() != ""]
    if len(parts) not in (2, 4):
        raise ValueError("Expect 2 or 4 comma-separated values.")

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
            theta_dot = np.deg2rad(theta_dot)
            psi_dot   = np.deg2rad(psi_dot)
        return theta, theta_dot, psi, psi_dot

def predict_ahead(x_now, u_now, tau):
    """
    First-order forward prediction for delay compensation over tau seconds.
    x_dot = A_c x + B_c u  =>  x(t+tau) ≈ x + tau * (A_c x + B_c u)
    """
    if tau <= 1e-6:
        return x_now
    return x_now + tau * (A_c @ x_now + B_c @ u_now)

def anti_windup_update(z, err, u_unsat, u_sat, dt, k_aw=1.0):
    """
    Back-calculation anti-windup: z_dot = err + k_aw*(u_sat - u_unsat)
    Here z is the 2D integral state corresponding to angle errors [theta, psi].
    """
    return z + (err + k_aw * (u_sat - u_unsat)) * dt

# =========================
# ========  GUI  ==========
# =========================
root = tk.Tk()
root.title("IoT Server Platform Dashboard")
root.geometry("1360x760")
root.minsize(1120, 680)  # ensure setpoint panel doesn't get cropped
root.configure(bg="#101820")

title_label = tk.Label(
    root, text="IoT SERVER PLATFORM",
    font=("Helvetica", 22, "bold"), bg="#101820", fg="#00ADEF"
)
title_label.pack(pady=8)

# Main two-pane layout so the left side is resizable
paned = ttk.Panedwindow(root, orient="horizontal")
paned.pack(fill="both", expand=True, padx=10, pady=10)

left_col = tk.Frame(paned, bg="#101820")
right_col = tk.Frame(paned, bg="#101820")
paned.add(left_col, weight=1)   # give left some weight
paned.add(right_col, weight=3)  # right gets more space for plots

# ---- Setpoints & robustness panel (TOP-LEFT)
sp_frame = tk.Frame(left_col, bg="#101820", highlightthickness=1, highlightbackground="#2A3A4D")
sp_frame.pack(fill="x", padx=6, pady=(0,8))

tk.Label(
    sp_frame, text="Control Panel", font=("Helvetica", 14, "bold"),
    bg="#101820", fg="#00ADEF"
).pack(anchor="w", padx=8, pady=(8, 2))

# Variables bound to widgets
theta_ref_deg_var = tk.DoubleVar(value=THETA_REF_DEG_INIT)
psi_ref_deg_var   = tk.DoubleVar(value=PSI_REF_DEG_INIT)
delay_ms_var      = tk.DoubleVar(value=60.0)   # estimate round-trip delay (ms)
slew_deg_s_var    = tk.DoubleVar(value=30.0)   # setpoint slew rate (deg/s)
smooth_alpha_var  = tk.DoubleVar(value=0.25)   # sensor smoothing alpha (0..1)
enable_pred_var   = tk.BooleanVar(value=True)  # enable predict-ahead
enable_aw_var     = tk.BooleanVar(value=True)  # enable anti-windup

# Display strings with 2-decimal formatting for setpoint labels
theta_ref_str = tk.StringVar(value=f"{THETA_REF_DEG_INIT:.2f}")
psi_ref_str   = tk.StringVar(value=f"{PSI_REF_DEG_INIT:.2f}")

def update_refs_from_gui(*_):
    """Update reference commands from GUI (deg->rad) and refresh 2-decimal labels."""
    global theta_ref_cmd, psi_ref_cmd
    th_deg = theta_ref_deg_var.get()
    ps_deg = psi_ref_deg_var.get()
    theta_ref_cmd = np.deg2rad(th_deg)
    psi_ref_cmd   = np.deg2rad(ps_deg)
    theta_ref_str.set(f"{th_deg:.2f}")
    psi_ref_str.set(f"{ps_deg:.2f}")

def reset_integrator():
    global z_int
    z_int = np.zeros(2)
    log_message("[INFO] Integral states reset.")

# Row: Pitch setpoint (with 2-dec label)
row1 = tk.Frame(sp_frame, bg="#101820"); row1.pack(fill="x", padx=8, pady=3)
tk.Label(row1, text="Pitch θᵣ (deg):", bg="#101820", fg="white", width=16, anchor="w").pack(side="left")
theta_scale = ttk.Scale(row1, from_=-60.0, to=60.0, variable=theta_ref_deg_var, command=lambda v: update_refs_from_gui())
theta_scale.pack(side="left", fill="x", expand=True, padx=8)
tk.Label(row1, textvariable=theta_ref_str, bg="#101820", fg="white", width=8, anchor="e").pack(side="left")

# Row: Yaw setpoint (with 2-dec label)
row2 = tk.Frame(sp_frame, bg="#101820"); row2.pack(fill="x", padx=8, pady=3)
tk.Label(row2, text="Yaw ψᵣ (deg):", bg="#101820", fg="white", width=16, anchor="w").pack(side="left")
psi_scale = ttk.Scale(row2, from_=-90.0, to=90.0, variable=psi_ref_deg_var, command=lambda v: update_refs_from_gui())
psi_scale.pack(side="left", fill="x", expand=True, padx=8)
tk.Label(row2, textvariable=psi_ref_str, bg="#101820", fg="white", width=8, anchor="e").pack(side="left")

# Row: Delay & smoothing
row3 = tk.Frame(sp_frame, bg="#101820"); row3.pack(fill="x", padx=8, pady=3)
tk.Label(row3, text="Delay (ms):", bg="#101820", fg="white", width=16, anchor="w").pack(side="left")
delay_entry = ttk.Entry(row3, textvariable=delay_ms_var, width=8); delay_entry.pack(side="left", padx=(4,16))
tk.Label(row3, text="Smooth α (0–1):", bg="#101820", fg="white", width=16, anchor="w").pack(side="left")
smooth_entry = ttk.Entry(row3, textvariable=smooth_alpha_var, width=8); smooth_entry.pack(side="left", padx=4)

# Row: Slew & toggles
row4 = tk.Frame(sp_frame, bg="#101820"); row4.pack(fill="x", padx=8, pady=3)
tk.Label(row4, text="Slew (deg/s):", bg="#101820", fg="white", width=16, anchor="w").pack(side="left")
slew_entry = ttk.Entry(row4, textvariable=slew_deg_s_var, width=8); slew_entry.pack(side="left", padx=(4,16))
ttk.Checkbutton(row4, text="Predict-ahead", variable=enable_pred_var).pack(side="left", padx=4)
ttk.Checkbutton(row4, text="Anti-windup", variable=enable_aw_var).pack(side="left", padx=8)

# Row: Buttons
btn_row = tk.Frame(sp_frame, bg="#101820"); btn_row.pack(fill="x", padx=8, pady=(6, 8))
ttk.Button(btn_row, text="Reset Integrator", command=reset_integrator).pack(side="left")

# initialize refs once
update_refs_from_gui()

# ---- Logs (BOTTOM-LEFT)
log_frame = tk.Frame(left_col, bg="#101820")
log_frame.pack(fill="both", expand=True, padx=6)

tk.Label(
    log_frame, text="Message Log", font=("Helvetica", 13, "bold"),
    bg="#101820", fg="white"
).pack(anchor="w")

msg_display = tk.Text(
    log_frame, height=12, bg="#1B2735", fg="white", font=("Consolas", 10), wrap="none"
)
msg_display.pack(fill="both", expand=True, pady=4)

tk.Label(
    log_frame, text="Transmitted Data Log (u0, u1)", font=("Helvetica", 13, "bold"),
    bg="#101820", fg="white"
).pack(anchor="w", pady=(6, 0))

data_display = tk.Text(
    log_frame, height=8, bg="#1B2735", fg="#00FF7F", font=("Consolas", 10), wrap="none"
)
data_display.pack(fill="x", pady=4)

def log_message(text):
    msg_display.insert(tk.END, text + "\n")
    msg_display.see(tk.END)

def log_data(text):
    data_display.insert(tk.END, text + "\n")
    data_display.see(tk.END)

# ---- Right plots (Transmitted vs Received)
fig = Figure(figsize=(9, 6), dpi=100)
fig.patch.set_facecolor("#101820")
ax_tx = fig.add_subplot(211)
ax_rx = fig.add_subplot(212)
for ax in (ax_tx, ax_rx):
    ax.set_facecolor("#1B2735")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

# 2-decimal tick formatters
fmt2 = FuncFormatter(lambda y, _: f"{y:.2f}")
ax_tx.yaxis.set_major_formatter(fmt2)
ax_rx.yaxis.set_major_formatter(fmt2)

ax_tx.set_title("Transmitted Control (u0, u1) [V]", color="white")
ax_rx.set_title("Received Angles (θ, ψ) [deg]", color="white")
ax_tx.set_ylabel("Value")
ax_rx.set_ylabel("Value")
ax_rx.set_xlabel("Samples")

canvas = FigureCanvasTkAgg(fig, master=right_col)
canvas.get_tk_widget().pack(side="right", fill="both", expand=True, padx=10, pady=6)

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
    global theta_d, psi_d, theta_filt, psi_filt, theta_dot_est, psi_dot_est
    global _last_t, _last_theta, _last_psi, u_last, z_int

    while True:
        try:
            msg = sub_socket.recv_string()
        except Exception as e:
            root.after(0, log_message, f"[RECV ERROR] {e}")
            continue

        t_now = time.monotonic()
        try:
            theta_raw, theta_dot_in, psi_raw, psi_dot_in = parse_sensor_message(msg)
        except Exception as e:
            root.after(0, log_message, f"[RECEIVED] {msg} (parse error: {e})")
            continue

        # Timing
        if _last_t is None:
            _last_t = t_now
        dt_meas = max(1e-4, t_now - _last_t)
        _last_t = t_now

        # Sensor smoothing on angles
        alpha = float(max(0.0, min(1.0, smooth_alpha_var.get())))
        theta_filt = (1 - alpha) * theta_filt + alpha * theta_raw
        psi_filt   = (1 - alpha) * psi_filt   + alpha * psi_raw

        # Derivative estimation if needed
        if _last_theta is None:
            _last_theta, _last_psi = theta_filt, psi_filt
        if theta_dot_in is None:
            theta_dot_est = 0.8 * theta_dot_est + 0.2 * ((theta_filt - _last_theta) / dt_meas)
            theta_dot = theta_dot_est
        else:
            theta_dot = theta_dot_in
            theta_dot_est = theta_dot_in
        if psi_dot_in is None:
            psi_dot_est = 0.8 * psi_dot_est + 0.2 * ((psi_filt - _last_psi) / dt_meas)
            psi_dot = psi_dot_est
        else:
            psi_dot = psi_dot_in
            psi_dot_est = psi_dot_in
        _last_theta, _last_psi = theta_filt, psi_filt

        # Slew-rate limit references (deg/s -> rad/s)
        slew_rad_s = np.deg2rad(max(0.0, float(slew_deg_s_var.get())))
        theta_d = ramp_to(theta_d, theta_ref_cmd, slew_rad_s, dt_meas)
        psi_d   = ramp_to(psi_d,   psi_ref_cmd,   slew_rad_s, dt_meas)
        x_ref = np.array([theta_d, 0.0, psi_d, 0.0])

        # Build current state
        x_now = np.array([theta_filt, theta_dot, psi_filt, psi_dot], dtype=float)

        # Predict-ahead for estimated network delay
        tau = max(0.0, float(delay_ms_var.get()) / 1000.0) if enable_pred_var.get() else 0.0
        x_for_ctrl = predict_ahead(x_now, u_last, tau)

        # Control law with integral action
        err_angles = np.array([x_for_ctrl[0] - x_ref[0], x_for_ctrl[2] - x_ref[2]])
        u_unsat = - K @ (x_for_ctrl - x_ref) - Ki * z_int
        u_sat = np.clip(u_unsat, -U_MAX, U_MAX)

        # Anti-windup
        if enable_aw_var.get():
            z_int = anti_windup_update(z_int, err_angles, u_unsat, u_sat, dt_meas, k_aw=1.0)
        else:
            for i in range(2):
                pushing_deeper = (abs(u_unsat[i]) > U_MAX) and (np.sign(u_unsat[i]) == np.sign(err_angles[i]*Ki[i]))
                if not pushing_deeper:
                    z_int[i] += err_angles[i] * dt_meas

        u = u_sat
        u0, u1 = float(u[0]), float(u[1])
        u_last = u.copy()

        # Publish control (u0,u1)
        out_msg = f"{u0:.4f},{u1:.4f}"
        try:
            pub_socket.send_string(out_msg)
        except Exception as e:
            root.after(0, log_message, f"[SEND ERROR] {e}")

        # GUI updates with 2-dec formatting for angles
        theta_deg = np.rad2deg(theta_filt)
        psi_deg   = np.rad2deg(psi_filt)
        root.after(0, log_message, f"[RECEIVED] {msg}")
        root.after(0, log_message, f"[ANGLES] θ={theta_deg:.2f}°, ψ={psi_deg:.2f}°")
        root.after(0, update_rx_graph, float(f"{theta_deg:.2f}"), float(f"{psi_deg:.2f}"))
        root.after(0, log_message, f"[SENT CTRL] {out_msg}")
        root.after(0, log_data,   f"{u0:7.3f}, {u1:7.3f}")
        root.after(0, update_tx_graph, u0, u1)

threading.Thread(target=receive_and_control, daemon=True).start()

# =========================
# ====== MAIN LOOP ========
# =========================
root.mainloop()
