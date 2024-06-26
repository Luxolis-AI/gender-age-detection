import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta

from matplotlib.ticker import MaxNLocator
# Use MaxNLocator for controlling y-axis tick intervals

def plot_data(filtered_detections):
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "figure.autolayout": True,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.title_fontsize": 14
    })

    if not filtered_detections:
        print("No data to plot.")
        return

    df = pd.DataFrame(filtered_detections, columns=["Timestamp", "Person ID", "Gender", "Age"])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Check if there is only one date, adjust to spread across more days for demonstration
    if df['Timestamp'].nunique() == 1:
        df['Timestamp'] += pd.to_timedelta(np.arange(len(df)), unit='D')

    plt.figure(figsize=(6, 4))
    df_daily = df.set_index('Timestamp').resample('D').size()
    if df_daily.empty:
        print("No data points to resample.")
        return

    df_daily = df_daily.astype(int)  # Ensure counts are integers
    ax = df_daily.plot(title='Total Detections Over Time', color='skyblue', linewidth=2.5)

    # Set y-axis to have integer ticks only, improving on previous settings
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
   

    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.savefig('/tmp/total_detections.png')
    plt.clf()

    # Define age categories if used later in code for demographic breakdowns
    age_categories = ['04 - 06', '07 - 08', '09 - 11', '12 - 19', '20 - 27', '28 - 35', '36 - 45', '46 - 60', '61 - 75']
    # Ensure proper plots for age categories if necessary
    if 'Gender' in df.columns and 'man' in df['Gender'].values:
        men_ages = df[df['Gender'] == 'man']['Age'].value_counts().reindex(age_categories, fill_value=0)
        men_ages.plot(kind='bar', color='lightblue', title='Men by Age')
        plt.savefig('/tmp/men_by_age.png')
        plt.clf()


    plt.figure(figsize=(6, 4))
    df[df['Gender'] == 'man']['Age'].value_counts().reindex(age_categories, fill_value=0).plot(kind='bar', color='lightblue', title='Men by Age')
    plt.xlabel("Age Group")
    plt.ylabel("Count")
    plt.savefig('/tmp/men_by_age.png')
    plt.clf()

    plt.figure(figsize=(6, 4))
    df[df['Gender'] == 'woman']['Age'].value_counts().reindex(age_categories, fill_value=0).plot(kind='bar', color='salmon', title='Women by Age')
    plt.xlabel("Age Group")
    plt.ylabel("Count")
    plt.savefig('/tmp/women_by_age.png')
    plt.clf()

    plt.figure(figsize=(6, 4))
    df['Gender'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightblue', 'salmon'], title='Ratio Between Men and Women')
    plt.ylabel('')
    plt.savefig('/tmp/ratio_between_men_and_women.png')
    plt.clf()

def display_graphs(graph_window, filtered_detections):
    # Load images and convert to PhotoImage
    img1 = ImageTk.PhotoImage(Image.open('/tmp/total_detections.png'))
    img2 = ImageTk.PhotoImage(Image.open('/tmp/men_by_age.png'))
    img3 = ImageTk.PhotoImage(Image.open('/tmp/women_by_age.png'))
    img4 = ImageTk.PhotoImage(Image.open('/tmp/ratio_between_men_and_women.png'))

    frame = tk.Frame(graph_window, bg="#000A1E")
    frame.pack(side="top", fill="both", expand=True)

    label1 = tk.Label(frame, image=img1, bg="#000A1E")
    label1.image = img1
    label1.grid(row=0, column=0, padx=10, pady=10)

    label2 = tk.Label(frame, image=img2, bg="#000A1E")
    label2.image = img2
    label2.grid(row=0, column=1, padx=10, pady=10)

    label3 = tk.Label(frame, image=img3, bg="#000A1E")
    label3.image = img3
    label3.grid(row=0, column=2, padx=10, pady=10)

    label4 = tk.Label(frame, image=img4, bg="#000A1E")
    label4.image = img4
    label4.grid(row=1, column=0, padx=10, pady=10)

    # Create a table in a frame
    table_frame = tk.Frame(frame, bg="#000A1E")
    table_frame.grid(row=1, column=1, columnspan=2, padx=10, pady=10, sticky="nsew")

    columns = ("Timestamp", "Person ID", "Gender", "Age")
    tree = ttk.Treeview(table_frame, columns=columns, show="headings")
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center")

    for detection in filtered_detections:
        tree.insert("", "end", values=detection)

    tree.pack(side="left", fill="both", expand=True)

    scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.pack(side="right", fill="y")

def filter_data(detections, period):
    now = datetime.now()
    if period == "Daily":
        return [d for d in detections if datetime.strptime(d[0], "%Y-%m-%d %H:%M:%S") > now - timedelta(days=1)]
    elif period == "Weekly":
        return [d for d in detections if datetime.strptime(d[0], "%Y-%m-%d %H:%M:%S") > now - timedelta(weeks=1)]
    elif period == "Monthly":
        return [d for d in detections if datetime.strptime(d[0], "%Y-%m-%d %H:%M:%S") > now - timedelta(days=30)]
    else:
        return detections

def show_graph(detections):
    graph_window = tk.Toplevel()
    graph_window.title("Detection Graphs")
    graph_window.geometry("1200x800")
    graph_window.configure(bg="#000A1E")

    filter_frame = tk.Frame(graph_window, bg="#000A1E")
    filter_frame.pack(side="top", fill="x")

    tk.Label(filter_frame, text="Filter:", bg="#000A1E", fg="#FFFFFF", font=("Arial", 12)).pack(side="left")
    tk.Button(filter_frame, text="Daily", command=lambda: plot_and_display(detections, "Daily", graph_window), bg="#000A1E", fg="#FFFFFF").pack(side="left")
    tk.Button(filter_frame, text="Weekly", command=lambda: plot_and_display(detections, "Weekly", graph_window), bg="#000A1E", fg="#FFFFFF").pack(side="left")
    tk.Button(filter_frame, text="Monthly", command=lambda: plot_and_display(detections, "Monthly", graph_window), bg="#000A1E", fg="#FFFFFF").pack(side="left")
    tk.Button(filter_frame, text="All", command=lambda: plot_and_display(detections, "All", graph_window), bg="#000A1E", fg="#FFFFFF").pack(side="left")

    plot_and_display(detections, "All", graph_window)

def plot_and_display(detections, period, graph_window):
    filtered_detections = filter_data(detections, period)
    plot_data(filtered_detections)
    display_graphs(graph_window, filtered_detections)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    test_detections = [
        ["2023-05-01 12:00:00", 1, "man", "20 - 27"],
        ["2023-05-01 13:00:00", 2, "woman", "28 - 35"],
        ["2023-05-01 14:00:00", 3, "man", "36 - 45"],
        ["2023-05-01 15:00:00", 4, "woman", "46 - 60"],
        ["2023-05-02 12:00:00", 5, "man", "20 - 27"],
        ["2023-05-02 13:00:00", 6, "woman", "28 - 35"],
    ]
    show_graph(test_detections)
    root.mainloop()


