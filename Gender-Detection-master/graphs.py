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




# def plot_data(filtered_detections):
#     sns.set_style("whitegrid")
#     plt.rcParams.update({
#         "figure.autolayout": True,
#         "axes.titlesize": 16,
#         "axes.labelsize": 14,
#         "xtick.labelsize": 12,
#         "ytick.labelsize": 12,
#         "legend.fontsize": 12,
#         "legend.title_fontsize": 14
#     })

#     df = pd.DataFrame(filtered_detections, columns=["Timestamp", "Person ID", "Gender", "Age"])
#     df['Timestamp'] = pd.to_datetime(df['Timestamp'])
#     age_categories = ['04 - 06', '07 - 08', '09 - 11', '12 - 19', '20 - 27', '28 - 35', '36 - 45', '46 - 60', '61 - 75']

#     plt.figure(figsize=(6, 4))
#     df_daily = df.set_index('Timestamp').resample('D').size()
#     df_daily = df_daily.astype(int)  # Ensure counts are integers
#     df_daily.plot(title='Total Detections Over Time', color='skyblue', linewidth=2.5)
#     plt.xlabel("Date")
#     plt.ylabel("Count")
#     plt.savefig('/tmp/total_detections.png')
#     plt.clf()

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










# import matplotlib.pyplot as plt
# import pandas as pd
# import tkinter as tk
# from PIL import Image, ImageTk
# from datetime import datetime, timedelta
# import seaborn as sns

# def plot_data(filtered_detections):
#     # Convert detections to DataFrame
#     df = pd.DataFrame(filtered_detections, columns=["Timestamp", "Person ID", "Gender", "Age"])
#     df['Timestamp'] = pd.to_datetime(df['Timestamp'])

#     # Set the aesthetic style of the plots
#     sns.set_style("whitegrid")

#     # Customize matplotlib parameters
#     plt.rcParams.update({
#         "figure.autolayout": True,
#         "axes.titlesize": 16,
#         "axes.labelsize": 14,
#         "xtick.labelsize": 12,
#         "ytick.labelsize": 12,
#         "legend.fontsize": 12,
#         "legend.title_fontsize": 14
#     })

#     # Plotting Total detections over time
#     plt.figure(figsize=(6, 4), dpi=100)
#     df.set_index('Timestamp').resample('D').size().plot(title='Total Detections Over Time', color='skyblue', linewidth=2.5)
#     plt.xlabel("Date")
#     plt.ylabel("Count")
#     plt.savefig('/tmp/total_detections.png')
#     plt.clf()

#     # Men by age
#     plt.figure(figsize=(6, 4), dpi=100)
#     age_categories = ['04 - 06', '07 - 08', '09 - 11', '12 - 19', '20 - 27', '28 - 35', '36 - 45', '46 - 60', '61 - 75']
#     df[df['Gender'] == 'man']['Age'].value_counts().reindex(age_categories).plot(kind='bar', color='lightblue', title='Men by Age')
#     plt.xlabel("Age Group")
#     plt.ylabel("Count")
#     plt.savefig('/tmp/men_by_age.png')
#     plt.clf()

#     # Women by age
#     plt.figure(figsize=(6, 4), dpi=100)
#     df[df['Gender'] == 'woman']['Age'].value_counts().reindex(age_categories).plot(kind='bar', color='salmon', title='Women by Age')
#     plt.xlabel("Age Group")
#     plt.ylabel("Count")
#     plt.savefig('/tmp/women_by_age.png')
#     plt.clf()

#     # Ratio between men and women
#     plt.figure(figsize=(6, 4), dpi=100)
#     df['Gender'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightblue', 'salmon'], title='Ratio Between Men and Women')
#     plt.ylabel('')  # Hide y-label for pie chart
#     plt.savefig('/tmp/ratio_between_men_and_women.png')
#     plt.clf()

# def display_graphs(graph_window):
#     img1 = ImageTk.PhotoImage(Image.open('/tmp/total_detections.png'))
#     img2 = ImageTk.PhotoImage(Image.open('/tmp/men_by_age.png'))
#     img3 = ImageTk.PhotoImage(Image.open('/tmp/women_by_age.png'))
#     img4 = ImageTk.PhotoImage(Image.open('/tmp/ratio_between_men_and_women.png'))

#     frame1 = tk.Frame(graph_window, bg="#000A1E")
#     frame1.pack(side="top", fill="both", expand=True)

#     frame2 = tk.Frame(graph_window, bg="#000A1E")
#     frame2.pack(side="top", fill="both", expand=True)

#     label1 = tk.Label(frame1, image=img1, bg="#000A1E")
#     label1.pack(side="left", expand=True)
#     label1.image = img1  # Keep a reference to avoid garbage collection

#     label2 = tk.Label(frame1, image=img2, bg="#000A1E")
#     label2.pack(side="left", expand=True)
#     label2.image = img2

#     label3 = tk.Label(frame1, image=img3, bg="#000A1E")
#     label3.pack(side="left", expand=True)
#     label3.image = img3

#     label4 = tk.Label(frame2, image=img4, bg="#000A1E")
#     label4.pack(side="left", expand=True)
#     label4.image = img4

#     print("Graphs displayed successfully")  # Debug print

# def filter_data(detections, period):
#     now = datetime.now()
#     if period == "Daily":
#         filtered_detections = [d for d in detections if datetime.strptime(d[0], "%Y-%m-%d %H:%M:%S") > now - timedelta(days=1)]
#     elif period == "Weekly":
#         filtered_detections = [d for d in detections if datetime.strptime(d[0], "%Y-%m-%d %H:%M:%S") > now - timedelta(weeks=1)]
#     elif period == "Monthly":
#         filtered_detections = [d for d in detections if datetime.strptime(d[0], "%Y-%m-%d %H:%M:%S") > now - timedelta(days=30)]
#     else:
#         filtered_detections = detections
#     return filtered_detections

# def show_graph(detections):
#     graph_window = tk.Toplevel()
#     graph_window.title("Detection Graphs")
#     graph_window.geometry("1200x800")
#     graph_window.configure(bg="#000A1E")

#     filter_frame = tk.Frame(graph_window, bg="#000A1E")
#     filter_frame.pack(side="top", fill="x")

#     tk.Label(filter_frame, text="Filter:", bg="#000A1E", fg="#FFFFFF", font=("Arial", 12)).pack(side="left")
#     tk.Button(filter_frame, text="Daily", command=lambda: plot_and_display(detections, "Daily", graph_window), bg="#000A1E", fg="#FFFFFF").pack(side="left")
#     tk.Button(filter_frame, text="Weekly", command=lambda: plot_and_display(detections, "Weekly", graph_window), bg="#000A1E", fg="#FFFFFF").pack(side="left")
#     tk.Button(filter_frame, text="Monthly", command=lambda: plot_and_display(detections, "Monthly", graph_window), bg="#000A1E", fg="#FFFFFF").pack(side="left")
#     tk.Button(filter_frame, text="All", command=lambda: plot_and_display(detections, "All", graph_window), bg="#000A1E", fg="#FFFFFF").pack(side="left")

#     plot_and_display(detections, "All", graph_window)

# def plot_and_display(detections, period, graph_window):
#     filtered_detections = filter_data(detections, period)
#     plot_data(filtered_detections)
#     display_graphs(graph_window)

# # Test function to see if plotting and displaying works
# if __name__ == "__main__":
#     root = tk.Tk()
#     root.withdraw()  # Hide the main root window
#     test_detections = [
#         ["2023-05-01 12:00:00", 1, "man", "20 - 27"],
#         ["2023-05-01 13:00:00", 2, "woman", "28 - 35"],
#         ["2023-05-01 14:00:00", 3, "man", "36 - 45"],
#         ["2023-05-01 15:00:00", 4, "woman", "46 - 60"],
#         ["2023-05-02 12:00:00", 5, "man", "20 - 27"],
#         ["2023-05-02 13:00:00", 6, "woman", "28 - 35"],
#     ]
#     show_graph(test_detections)
#     root.mainloop()














# import sys
# import pandas as pd
# from PyQt5 import QtWidgets, QtCore
# import pyqtgraph as pg
# from datetime import datetime, timedelta

# class DetectionApp(QtWidgets.QMainWindow):
#     def __init__(self, detections):
#         super().__init__()
#         self.detections = detections
#         self.setWindowTitle("Detection Graphs")
#         self.setGeometry(100, 100, 1200, 800)

#         self.main_widget = QtWidgets.QWidget(self)
#         self.setCentralWidget(self.main_widget)
#         self.layout = QtWidgets.QVBoxLayout(self.main_widget)

#         self.filter_frame = QtWidgets.QFrame(self.main_widget)
#         self.filter_layout = QtWidgets.QHBoxLayout(self.filter_frame)
#         self.layout.addWidget(self.filter_frame)

#         self.filter_label = QtWidgets.QLabel("Filter:")
#         self.filter_layout.addWidget(self.filter_label)

#         self.daily_button = QtWidgets.QPushButton("Daily")
#         self.daily_button.clicked.connect(lambda: self.plot_and_display("Daily"))
#         self.filter_layout.addWidget(self.daily_button)

#         self.weekly_button = QtWidgets.QPushButton("Weekly")
#         self.weekly_button.clicked.connect(lambda: self.plot_and_display("Weekly"))
#         self.filter_layout.addWidget(self.weekly_button)

#         self.monthly_button = QtWidgets.QPushButton("Monthly")
#         self.monthly_button.clicked.connect(lambda: self.plot_and_display("Monthly"))
#         self.filter_layout.addWidget(self.monthly_button)

#         self.all_button = QtWidgets.QPushButton("All")
#         self.all_button.clicked.connect(lambda: self.plot_and_display("All"))
#         self.filter_layout.addWidget(self.all_button)

#         self.scroll_area = QtWidgets.QScrollArea(self.main_widget)
#         self.scroll_area.setWidgetResizable(True)
#         self.layout.addWidget(self.scroll_area)

#         self.scroll_widget = QtWidgets.QWidget()
#         self.scroll_layout = QtWidgets.QVBoxLayout(self.scroll_widget)
#         self.scroll_area.setWidget(self.scroll_widget)

#         self.plot_and_display("All")

#     def filter_data(self, period):
#         now = datetime.now()
#         if period == "Daily":
#             filtered_detections = [d for d in self.detections if datetime.strptime(d[0], "%Y-%m-%d %H:%M:%S") > now - timedelta(days=1)]
#         elif period == "Weekly":
#             filtered_detections = [d for d in self.detections if datetime.strptime(d[0], "%Y-%m-%d %H:%M:%S") > now - timedelta(weeks=1)]
#         elif period == "Monthly":
#             filtered_detections = [d for d in self.detections if datetime.strptime(d[0], "%Y-%m-%d %H:%M:%S") > now - timedelta(days=30)]
#         else:
#             filtered_detections = self.detections
#         return filtered_detections

#     def plot_data(self, filtered_detections):
#         df = pd.DataFrame(filtered_detections, columns=["Timestamp", "Person ID", "Gender", "Age"])
#         df['Timestamp'] = pd.to_datetime(df['Timestamp'])

#         plots = []

#         # Total detections over time
#         plot_widget = pg.PlotWidget(title='Total Detections Over Time')
#         total_detections = df.set_index('Timestamp').resample('D').size()
#         plot_widget.plot(total_detections.index.to_pydatetime(), total_detections.values, pen='b')
#         plots.append(plot_widget)

#         # Gender distribution
#         plot_widget = pg.PlotWidget(title='Gender Distribution')
#         gender_counts = df['Gender'].value_counts()
#         bar_graph = pg.BarGraphItem(x=gender_counts.index, height=gender_counts.values, width=0.3, brush='b')
#         plot_widget.addItem(bar_graph)
#         plots.append(plot_widget)

#         # Age distribution
#         plot_widget = pg.PlotWidget(title='Age Distribution')
#         age_counts = df['Age'].value_counts()
#         bar_graph = pg.BarGraphItem(x=age_counts.index, height=age_counts.values, width=0.3, brush='b')
#         plot_widget.addItem(bar_graph)
#         plots.append(plot_widget)

#         # Number of people detected by gender
#         plot_widget = pg.PlotWidget(title='Number of People Detected by Gender')
#         gender_counts = df['Gender'].value_counts()
#         pie_chart = pg.PieChartItem(labels=gender_counts.index, sizes=gender_counts.values, brush='b')
#         plot_widget.addItem(pie_chart)
#         plots.append(plot_widget)

#         # Detection by age group (men and women)
#         plot_widget = pg.PlotWidget(title='Detection by Age Group (Men and Women)')
#         age_gender_counts = df.groupby(['Age', 'Gender']).size().unstack()
#         bar_graph_men = pg.BarGraphItem(x=age_gender_counts.index, height=age_gender_counts['Male'], width=0.3, brush='b')
#         bar_graph_women = pg.BarGraphItem(x=age_gender_counts.index, height=age_gender_counts['Female'], width=0.3, brush='r')
#         plot_widget.addItem(bar_graph_men)
#         plot_widget.addItem(bar_graph_women)
#         plots.append(plot_widget)

#         return plots

#     def plot_and_display(self, period):
#         filtered_detections = self.filter_data(period)
#         plots = self.plot_data(filtered_detections)

#         for i in reversed(range(self.scroll_layout.count())):
#             widget = self.scroll_layout.itemAt(i).widget()
#             if widget:
#                 widget.setParent(None)

#         for plot in plots:
#             self.scroll_layout.addWidget(plot)

# if __name__ == "__main__":
#     detections = [
#         ["2024-05-28 12:00:00", 1, "Male", 25],
#         ["2024-05-28 12:05:00", 2, "Female", 30],
#         ["2024-05-27 12:00:00", 3, "Male", 35],
#         ["2024-05-26 12:00:00", 4, "Female", 40],
#         # Add more sample data as needed
#     ]

#     app = QtWidgets.QApplication(sys.argv)
#     main_window = DetectionApp(detections)
#     main_window.show()
#     sys.exit(app.exec_())






# import matplotlib.pyplot as plt
# import pandas as pd
# import tkinter as tk
# from tkinter import ttk
# from PIL import Image, ImageTk
# from datetime import datetime, timedelta

# def plot_data(filtered_detections):
#     df = pd.DataFrame(filtered_detections, columns=["Timestamp", "Person ID", "Gender", "Age"])
#     df['Timestamp'] = pd.to_datetime(df['Timestamp'])

#     # Total detections over time
#     plt.figure(figsize=(10, 6))
#     df.set_index('Timestamp').resample('D').size().plot(title='Total Detections Over Time')
#     plt.savefig('/tmp/total_detections.png')
#     plt.clf()

#     # Gender distribution
#     plt.figure(figsize=(10, 6))
#     df['Gender'].value_counts().plot(kind='bar', title='Gender Distribution')
#     plt.savefig('/tmp/gender_distribution.png')
#     plt.clf()

#     # Age distribution
#     plt.figure(figsize=(10, 6))
#     df['Age'].value_counts().plot(kind='bar', title='Age Distribution')
#     plt.savefig('/tmp/age_distribution.png')
#     plt.clf()

#     # Number of people detected by gender
#     plt.figure(figsize=(10, 6))
#     df['Gender'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, title='Number of People Detected by Gender')
#     plt.savefig('/tmp/people_detected_by_gender.png')
#     plt.clf()

#     # Detection by age group (men and women)
#     plt.figure(figsize=(10, 6))
#     df.groupby(['Age', 'Gender']).size().unstack().plot(kind='bar', stacked=True, title='Detection by Age Group (Men and Women)')
#     plt.savefig('/tmp/detection_by_age_group.png')
#     plt.clf()

# def display_graphs(scrollable_frame):
#     img1 = ImageTk.PhotoImage(Image.open('/tmp/total_detections.png'))
#     img2 = ImageTk.PhotoImage(Image.open('/tmp/gender_distribution.png'))
#     img3 = ImageTk.PhotoImage(Image.open('/tmp/age_distribution.png'))
#     img4 = ImageTk.PhotoImage(Image.open('/tmp/people_detected_by_gender.png'))
#     img5 = ImageTk.PhotoImage(Image.open('/tmp/detection_by_age_group.png'))

#     tk.Label(scrollable_frame, image=img1).pack(side="top", fill="both", expand="yes")
#     tk.Label(scrollable_frame, image=img2).pack(side="top", fill="both", expand="yes")
#     tk.Label(scrollable_frame, image=img3).pack(side="top", fill="both", expand="yes")
#     tk.Label(scrollable_frame, image=img4).pack(side="top", fill="both", expand="yes")
#     tk.Label(scrollable_frame, image=img5).pack(side="top", fill="both", expand="yes")

#     # Keep a reference to avoid garbage collection
#     scrollable_frame.image1 = img1
#     scrollable_frame.image2 = img2
#     scrollable_frame.image3 = img3
#     scrollable_frame.image4 = img4
#     scrollable_frame.image5 = img5

# def filter_data(detections, period):
#     now = datetime.now()
#     if period == "Daily":
#         filtered_detections = [d for d in detections if datetime.strptime(d[0], "%Y-%m-%d %H:%M:%S") > now - timedelta(days=1)]
#     elif period == "Weekly":
#         filtered_detections = [d for d in detections if datetime.strptime(d[0], "%Y-%m-%d %H:%M:%S") > now - timedelta(weeks=1)]
#     elif period == "Monthly":
#         filtered_detections = [d for d in detections if datetime.strptime(d[0], "%Y-%m-%d %H:%M:%S") > now - timedelta(days=30)]
#     else:
#         filtered_detections = detections
#     return filtered_detections

# def show_graph(detections):
#     graph_window = tk.Toplevel()
#     graph_window.title("Detection Graphs")
#     graph_window.geometry("1200x800")
#     graph_window.configure(bg="#000A1E")

#     container = ttk.Frame(graph_window)
#     canvas = tk.Canvas(container, bg="#000A1E")
#     scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
#     scrollable_frame = ttk.Frame(canvas)

#     scrollable_frame.bind(
#         "<Configure>",
#         lambda e: canvas.configure(
#             scrollregion=canvas.bbox("all")
#         )
#     )

#     canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
#     canvas.configure(yscrollcommand=scrollbar.set)

#     container.pack(fill="both", expand=True)
#     canvas.pack(side="left", fill="both", expand=True)
#     scrollbar.pack(side="right", fill="y")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            

#     filter_frame = tk.Frame(scrollable_frame, bg="#000A1E")
#     filter_frame.pack(side="top", fill="x")

#     tk.Label(filter_frame, text="Filter:", bg="#000A1E", fg="#FFFFFF", font=("Arial", 12)).pack(side="left")
#     tk.Button(filter_frame, text="Daily", command=lambda: plot_and_display(detections, "Daily", scrollable_frame), bg="#000A1E", fg="#FFFFFF").pack(side="left")
#     tk.Button(filter_frame, text="Weekly", command=lambda: plot_and_display(detections, "Weekly", scrollable_frame), bg="#000A1E", fg="#FFFFFF").pack(side="left")
#     tk.Button(filter_frame, text="Monthly", command=lambda: plot_and_display(detections, "Monthly", scrollable_frame), bg="#000A1E", fg="#FFFFFF").pack(side="left")
#     tk.Button(filter_frame, text="All", command=lambda: plot_and_display(detections, "All", scrollable_frame), bg="#000A1E", fg="#FFFFFF").pack(side="left")

#     plot_and_display(detections, "All", scrollable_frame)

# def plot_and_display(detections, period, scrollable_frame):
#     filtered_detections = filter_data(detections, period)
#     plot_data(filtered_detections)
#     display_graphs(scrollable_frame)

