"""TensorKernel demo runner.

This script imports the TensorKernel Python binding and displays a simple
Tkinter UI to run a polished tensor operation demo.
"""

import sys
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import messagebox

try:
    import numpy as np
    import tensorkernel as tk_module
except ImportError as exc:
    print("ERROR: Failed to import required modules.")
    print(exc)
    print("\nIf `tensorkernel` is not installed, build the project first and install it into your Python environment.")
    print("Example build path: create a CMake build, compile the pybind11 extension, then install it.")
    sys.exit(1)


def format_matrix(matrix, precision=4):
    return np.array2string(matrix, precision=precision, floatmode="fixed", suppress_small=True)


def section_header(title):
    line = "=" * len(title)
    return f"{title}\n{line}\n"


def subsection_title(title):
    line = "-" * len(title)
    return f"{title}\n{line}\n"


def build_demo_output():
    lines = []
    lines.append(section_header("TensorKernel GUI Demo"))

    A = tk_module.Tensor.rand([4, 4])
    B = tk_module.Tensor.rand([4, 4])
    C = tk_module.Tensor.zeros([4, 4])

    lines.append(subsection_title("1) Create tensors"))
    lines.append(f"A shape: {A.shape()}")
    lines.append(f"B shape: {B.shape()}")
    lines.append(f"C shape: {C.shape()}\n")
    lines.append("A =")
    lines.append(format_matrix(A.numpy()))
    lines.append("\nB =")
    lines.append(format_matrix(B.numpy()))
    lines.append("\nC (initial zeros) =")
    lines.append(format_matrix(C.numpy()))

    lines.append(subsection_title("2) Matrix multiplication"))
    tk_module.cpu.matmul_simd(A, B, C)
    lines.append("Operation: C = A @ B")
    lines.append(format_matrix(C.numpy()))

    lines.append(subsection_title("3) Add ones"))
    D = tk_module.Tensor.ones([4, 4])
    tk_module.cpu.add_simd(C, D, C)
    lines.append("Operation: C += 1")
    lines.append(format_matrix(C.numpy()))

    lines.append(subsection_title("4) ReLU activation"))
    tk_module.cpu.relu_simd(C)
    lines.append("Operation: C = relu(C)")
    lines.append(format_matrix(C.numpy()))

    lines.append(subsection_title("5) Reduction & verification"))
    total = tk_module.cpu.reduce_sum_simd(C)
    a_np = A.numpy()
    b_np = B.numpy()
    expected = a_np @ b_np + np.ones((4, 4), dtype=np.float32)
    expected = np.maximum(expected, 0.0)
    diff = np.max(np.abs(C.numpy() - expected))

    lines.append(f"Sum of all values in C: {total:.6f}")
    lines.append(f"Max difference vs NumPy expected: {diff:.6e}")
    if diff < 1e-6:
        lines.append("\n✅ Verification passed: TensorKernel output matches NumPy expected values.")
    else:
        lines.append("\n⚠️ Verification warning: TensorKernel output differs from NumPy expected values.")

    lines.append(section_header("Demo complete"))
    return "\n".join(lines)


def run_demo(output_widget, run_button, status_label):
    def task():
        try:
            output_text = build_demo_output()
            output_widget.after(0, lambda: output_widget.configure(state="normal"))
            output_widget.after(0, lambda: output_widget.delete("1.0", tk.END))
            output_widget.after(0, lambda: output_widget.insert(tk.END, output_text))
            output_widget.after(0, lambda: output_widget.configure(state="disabled"))
            output_widget.after(0, lambda: status_label.config(text="Ready", foreground="#0A7D00"))
        except Exception as exc:
            output_widget.after(0, lambda: messagebox.showerror("Demo error", str(exc)))
            output_widget.after(0, lambda: status_label.config(text="Error", foreground="#A00000"))
        finally:
            output_widget.after(0, lambda: run_button.configure(state="normal"))

    run_button.configure(state="disabled")
    status_label.config(text="Running demo...", foreground="#0055AA")
    threading.Thread(target=task, daemon=True).start()


def create_ui():
    root = tk.Tk()
    root.title("TensorKernel Demo")
    root.geometry("900x780")

    content = ttk.Frame(root, padding=16)
    content.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    title_label = ttk.Label(content, text="TensorKernel GUI Demo", font=("Segoe UI", 18, "bold"))
    title_label.grid(row=0, column=0, sticky="w")

    subtitle = ttk.Label(content, text="Click Run Demo to execute the TensorKernel pipeline and display results.", font=("Segoe UI", 11))
    subtitle.grid(row=1, column=0, sticky="w", pady=(8, 16))

    output_text = scrolledtext.ScrolledText(content, wrap=tk.WORD, font=("Consolas", 11), state="disabled")
    output_text.grid(row=2, column=0, sticky="nsew")
    content.rowconfigure(2, weight=1)
    content.columnconfigure(0, weight=1)

    bottom_frame = ttk.Frame(content)
    bottom_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
    bottom_frame.columnconfigure(1, weight=1)

    run_button = ttk.Button(bottom_frame, text="Run Demo", command=lambda: run_demo(output_text, run_button, status_label))
    run_button.grid(row=0, column=0, sticky="w")

    status_label = ttk.Label(bottom_frame, text="Ready", font=("Segoe UI", 10, "italic"))
    status_label.grid(row=0, column=1, sticky="e")

    return root


def main():
    root = create_ui()
    root.mainloop()


if __name__ == "__main__":
    main()
