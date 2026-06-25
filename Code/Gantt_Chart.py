import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

tasks = [
    "Detection Model Creation",
    "Hardware Setup",
    "Data Collection",
    "Spoofing Generator",
    "GUI Development",
    "Update Model",
    "Testing & Documentation"
]

start_dates = [
    "01-10-2025",
    "15-12-2025",
    "01-02-2026",
    "01-04-2026",
    "01-04-2026",
    "01-05-2026",
    "01-05-2026"
]

durations = [
    75,  # Detection Model Creation (Oct-Dec)
    60,  # Hardware Setup (Dec-Feb)
    60,  # Data Collection (Feb-Apr)
    30,  # Spoofing Generator (Apr-May)
    30,  # GUI (Apr-May)
    31,  # Update Model (May-Jun)
    31   # Testing (May-Jun)
]


start_dates = [
    datetime.strptime(d, "%d-%m-%Y")
    for d in start_dates
]

df = pd.DataFrame({
    "Task": tasks,
    "Start": start_dates,
    "Duration": durations
})


plt.figure(figsize=(14,7))

plt.title(
    "GPS Spoofing Detection Project Gantt Chart",
    fontsize=14
)

for i, task in enumerate(df["Task"]):
    plt.barh(
        i,
        df["Duration"][i],
        left=df["Start"][i],
        height=0.5,
        color="#7EC8E3"
    )


plt.yticks(
    range(len(tasks)),
    tasks
)

plt.xlabel("Timeline")
plt.ylabel("Tasks")

plt.xlim(
    datetime(2025,10,1),
    datetime(2026,6,30)
)

plt.grid(
    axis="x",
    linestyle="--",
    alpha=0.3
)

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()