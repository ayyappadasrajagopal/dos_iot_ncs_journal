import zmq
import threading
import time
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import random

# ========== ZMQ CONFIGURATION ==========
MY_IP = "10.128.6.217"
MY_PORT = 5557

PEER_IP = "10.128.7.67"
PEER_PORT = 5555

context = zmq.Context()

# PUB socket
pub_socket = context.socket(zmq.PUB)
pub_socket.bind(f"tcp://{MY_IP}:{MY_PORT}")

# SUB socket
sub_socket = context.socket(zmq.SUB)
sub_socket.connect(f"tcp://{PEER_IP}:{PEER_PORT}")
sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # subscribe to all

# ========== TKINTER GUI ==========
root = tk.Tk()
root.title("IoT Server Platform Dashboard")
root.geometry("1250x700")
root.configure(bg="#101820")

title_label = tk.Label(
    root,
    text="IoT SERVER PLATFORM",
    font=("Helvetica", 22, "bold"),
    bg="#101820",
    fg="#00ADEF"
)
title_label.pack(pady=10)

# ====== MAIN LAYOUT: LEFT PANEL (LOGS) + RIGHT PANEL (GRAPH) ======
main_frame = tk.Frame(root, bg="#101820")
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

# LEFT SIDE PANEL - LOGS
log_frame = tk.Frame(main_frame, bg="#101820")
log_frame.pack(side="left", fill="y", padx=10)

# Message log (received/sent info)
tk.Label(
    log_frame,
    text="Message Log:",
    font=("Helvetica", 14, "bold"),
    bg="#101820",
    fg="white"
).pack(anchor="w")

msg_display = tk.Text(
    log_frame,
    height=18,
    width=45,
    bg="#1B2735",
    fg="white",
    font=("Consolas", 11)
)
msg_display.pack(pady=5)

# Transmitted numeric data log
tk.Label(
    log_frame,
    text="Transmitted Data Log:",
    font=("Helvetica", 14, "bold"),
    bg="#101820",
    fg="white"
).pack(anchor="w", pady=(20, 0))

data_display = tk.Text(
    log_frame,
    height=15,
    width=45,
    bg="#1B2735",
    fg="#00FF7F",
    font=("Consolas", 11)
)
data_display.pack(pady=5)

def log_message(text):
    msg_display.insert(tk.END, text + "\n")
    msg_display.see(tk.END)

def log_data(text):
    data_display.insert(tk.END, text + "\n")
    data_display.see(tk.END)

# ========== MATPLOTLIB SETUP ==========
fig = Figure(figsize=(9, 6), dpi=100)
fig.patch.set_facecolor("#101820")

ax_tx = fig.add_subplot(211)
ax_rx = fig.add_subplot(212)

for ax in [ax_tx, ax_rx]:
    ax.set_facecolor("#1B2735")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

ax_tx.set_title("Transmitted Data", color="white")
ax_rx.set_title("Received Data", color="white")
ax_tx.set_ylabel("Value")
ax_rx.set_ylabel("Value")
ax_rx.set_xlabel("Samples")

canvas = FigureCanvasTkAgg(fig, master=main_frame)
canvas.get_tk_widget().pack(side="right", fill="both", expand=True, padx=10)

# Data storage
x_data_tx, y1_tx, y2_tx = [], [], []
x_data_rx, y1_rx, y2_rx = [], [], []

# Initialize line objects
line_tx1, = ax_tx.plot([], [], color="#00ADEF", label="Value 1", linewidth=2)
line_tx2, = ax_tx.plot([], [], color="#FF6F61", label="Value 2", linewidth=2)
line_rx1, = ax_rx.plot([], [], color="#00FF7F", label="Value 1", linewidth=2)
line_rx2, = ax_rx.plot([], [], color="#FFBF00", label="Value 2", linewidth=2)

ax_tx.legend(facecolor="#1B2735", labelcolor="white")
ax_rx.legend(facecolor="#1B2735", labelcolor="white")
canvas.draw()

# ========== GRAPH UPDATE FUNCTIONS ==========
max_y_tx, min_y_tx = None, None
max_y_rx, min_y_rx = None, None

def update_transmit_graph(msg):
    global max_y_tx, min_y_tx
    try:
        parts = [float(x.strip()) for x in msg.split(",")]
    except:
        return

    x_data_tx.append(time.strftime("%H:%M:%S"))
    if len(parts) >= 1: y1_tx.append(parts[0])
    if len(parts) >= 2: y2_tx.append(parts[1])

    if len(x_data_tx) > 50:
        x_data_tx.pop(0)
        y1_tx.pop(0)
        y2_tx.pop(0)

    line_tx1.set_data(range(len(x_data_tx)), y1_tx)
    line_tx2.set_data(range(len(x_data_tx)), y2_tx)

    all_vals = y1_tx + y2_tx
    if all_vals:
        cmax, cmin = max(all_vals), min(all_vals)
        max_y_tx = cmax if max_y_tx is None else max(max_y_tx, cmax)
        min_y_tx = cmin if min_y_tx is None else min(min_y_tx, cmin)
        margin = (max_y_tx - min_y_tx) * 0.1 if max_y_tx != min_y_tx else 1
        ax_tx.set_ylim(min_y_tx - margin, max_y_tx + margin)

    ax_tx.set_xlim(0, len(x_data_tx))
    canvas.draw_idle()

def update_received_graph(msg):
    global max_y_rx, min_y_rx
    try:
        parts = [float(x.strip()) for x in msg.split(",")]
    except:
        return

    # only update receive arrays and graph
    x_data_rx.append(time.strftime("%H:%M:%S"))
    if len(parts) >= 1: y1_rx.append(parts[0])
    if len(parts) >= 2: y2_rx.append(parts[1])

    if len(x_data_rx) > 50:
        x_data_rx.pop(0)
        y1_rx.pop(0)
        y2_rx.pop(0)

    line_rx1.set_data(range(len(x_data_rx)), y1_rx)
    line_rx2.set_data(range(len(x_data_rx)), y2_rx)

    all_vals = y1_rx + y2_rx
    if all_vals:
        cmax, cmin = max(all_vals), min(all_vals)
        max_y_rx = cmax if max_y_rx is None else max(max_y_rx, cmax)
        min_y_rx = cmin if min_y_rx is None else min(min_y_rx, cmin)
        margin = (max_y_rx - min_y_rx) * 0.1 if max_y_rx != min_y_rx else 1
        ax_rx.set_ylim(min_y_rx - margin, max_y_rx + margin)

    ax_rx.set_xlim(0, len(x_data_rx))
    canvas.draw_idle()

# ========== RECEIVE THREAD ==========
def receive_messages():
    while True:
        msg = sub_socket.recv_string()
        # Ensure received messages don't trigger transmit graph
        log_message(f"[RECEIVED] {msg}")
        root.after(0, update_received_graph, msg)

threading.Thread(target=receive_messages, daemon=True).start()

# ========== AUTO-SEND SIMULATION THREAD ==========
def auto_send():
    while True:
        v1 = random.randint(-50, 80)
        v2 = random.randint(-100, 140)
        msg = f"{v1},{v2}"
        pub_socket.send_string(msg)  # broadcast to peer
        log_message(f"[SENT] {msg}")
        log_data(f"{v1:5.1f}, {v2:5.1f}")
        # Update only TX graph locally
        root.after(0, update_transmit_graph, msg)
        time.sleep(1)

threading.Thread(target=auto_send, daemon=True).start()

# ========== RUN GUI ==========
root.mainloop()
