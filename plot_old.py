import pandas as pd
import matplotlib.pyplot as plt

m = pd.read_csv("out/master_table.csv")

cell = 1
d = m[m["cell"] == cell].sort_values("cyc")

plt.figure()
plt.plot(d["cyc"], d["capacity_mAh"], marker="o")
plt.xlabel("Characterisation cycle (0,100,200,...)")
plt.ylabel("Capacity (mAh)")
plt.title(f"Cell {cell}: Capacity fade")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(d["cyc"], d["Tmax_C"], marker="o")
plt.xlabel("Characterisation cycle")
plt.ylabel("Tmax (°C)")
plt.title(f"Cell {cell}: Max temperature during C1dc")
plt.grid(True)
plt.show()