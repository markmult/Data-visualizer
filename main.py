import sys
import pandas as pd
from PySide2.QtCore import Qt, Slot
from PySide2.QtGui import QPainter
from PySide2.QtWidgets import (QAction, QApplication, QHeaderView, QHBoxLayout, QLabel, QLineEdit,
                               QMainWindow, QPushButton, QTableWidget, QTableWidgetItem, QDoubleSpinBox,
                               QVBoxLayout, QWidget, QGroupBox, QCheckBox, QSpinBox, QFormLayout)
from PySide2.QtCharts import QtCharts
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import transformations
import re

class Widget(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        # Store input data
        self.data = None

        ################### UI Elements ####################
        #################### Left side ####################
        self.file_name = QLineEdit()
        self.load = QPushButton("Load file")
        self.load.setStyleSheet("background-color:#eb8c34;")

        self.table = QTableWidget()

        self.drop_cols = QLineEdit()
        self.drop = QPushButton("Drop Columns")
        self.drop.setStyleSheet("background-color:#eb8c34;")

        self.index_col = QLineEdit()
        self.index = QPushButton("Set column to index")
        self.index.setStyleSheet("background-color:#eb8c34;")

        self.quit = QPushButton("Quit")
        self.quit.setStyleSheet("background-color:#eb8c34;")

        self.plot = QPushButton("Plot (After you have loaded the data and selected correct options)")
        self.plot.setStyleSheet("background-color:#eb8c34;")

        self.color_plot = QPushButton("Plot with colors (Based on the index column value)")
        self.color_plot.setStyleSheet("background-color:#eb8c34;")

        # Disabling 'Load' button
        self.load.setEnabled(False)

        self.right = QVBoxLayout()
        self.right.setMargin(10)
        self.right.addWidget(QLabel("File path"))
        self.right.addWidget(self.file_name)
        self.right.addWidget(self.load)
        self.right.addWidget(self.table)
        self.right.addWidget(QLabel("Do you want to drop any columns? Type column numbers separated by comma eg. 0,1,5,8"))
        self.right.addWidget(self.drop_cols)
        self.right.addWidget(self.drop)
        self.right.addWidget(QLabel("Set column to index (Optional), type column name eg. City"))
        self.right.addWidget(self.index_col)
        self.right.addWidget(self.index)
        self.right.addWidget(QLabel("Please, drop data columns that contain only text before plotting. Index column is exception."))
        self.right.addWidget(self.plot)
        self.right.addWidget(self.color_plot)
        self.right.addWidget(self.quit)

        #################### Right side ####################
        self.left = QVBoxLayout()

        # Normalization / Scaling options
        self.scaling = QGroupBox("Scaling and normalization. Please select atleast two options")
        self.scaling.setFlat(True)
        self.minmax = QCheckBox("&Minmax")
        self.norm = QCheckBox("&Standard scaler")
        self.mean = QCheckBox("&Mean normalization")
        self.vect = QCheckBox("&Normalizer (unit vector normalization)")
        self.robust = QCheckBox("&Robust scaler")
        self.vbox1 = QVBoxLayout()
        self.vbox1.addWidget(self.minmax)
        self.vbox1.addWidget(self.norm)
        self.vbox1.addWidget(self.mean)
        self.vbox1.addWidget(self.vect)
        self.vbox1.addWidget(self.robust)
        self.scaling.setLayout(self.vbox1)
        self.left.addWidget(self.scaling)

        # Dimensionality reduction options
        self.dim_red = QGroupBox("Dimensionality reduction. Please select atleast two options")
        self.dim_red.setFlat(True)
        self.pca = QCheckBox("&PCA")
        self.mds = QCheckBox("&MDS")
        self.isomap = QCheckBox("&Isomap")
        self.umap = QCheckBox("&UMAP")
        self.t_sne = QCheckBox("&t-SNE")

        self.vbox2 = QVBoxLayout()
        self.vbox2.addWidget(self.pca)
        self.vbox2.addWidget(self.mds)
        self.vbox2.addWidget(self.isomap)
        self.vbox2.addWidget(self.umap)
        self.vbox2.addWidget(self.t_sne)
        self.dim_red.setLayout(self.vbox2)

        # Parameter selection 
        self.formGroupBox = QGroupBox("Parameters")
        self.isomap_neighbors = QSpinBox()
        self.isomap_neighbors.setRange(2, 30)
        self.isomap_neighbors.setValue(5)

        self.umap_neigbors = QSpinBox()
        self.umap_neigbors.setRange(2, 50)
        self.umap_neigbors.setValue(15)

        self.umap_dist = QDoubleSpinBox()
        self.umap_dist.setSingleStep(0.025)
        self.umap_dist.setRange(0.0, 0.99)
        self.umap_dist.setValue(0.1)

        self.tsne_neighbors = QSpinBox()
        self.tsne_neighbors.setRange(5, 50)
        self.tsne_neighbors.setValue(30)

        l1 = QLabel("Isomap n_neighbors:")
        l2 = QLabel("Umap n_neighbors:")
        l3 = QLabel("Umap min_distance:")
        l4 = QLabel("t-SNE perplexity:")

        # Options to form layout
        self.formlayout = QFormLayout()
        self.formlayout.addRow(l1, self.isomap_neighbors)
        self.formlayout.addRow(l2, self.umap_neigbors)
        self.formlayout.addRow(l3, self.umap_dist)
        self.formlayout.addRow(l4, self.tsne_neighbors)

        self.formGroupBox.setLayout(self.formlayout)

        self.left.addWidget(self.dim_red)
        self.left.addWidget(self.formGroupBox)

        # QWidget Layout
        self.layout = QHBoxLayout()

        #self.table_view.setSizePolicy(size)
        self.layout.addLayout(self.right)
        self.layout.addLayout(self.left)

        # Set the layout to the QWidget
        self.setLayout(self.layout)

        # Connect buttons with functions
        self.quit.clicked.connect(self.quit_application)
        self.plot.clicked.connect(self.plot_data)
        self.color_plot.clicked.connect(self.plot_colors)
        self.file_name.textChanged[str].connect(self.check_disable)
        self.load.clicked.connect(self.load_data)
        self.drop.clicked.connect(self.drop_column)
        self.index.clicked.connect(self.change_index)

        self.minmax.stateChanged.connect(self.plot_disable)
        self.norm.stateChanged.connect(self.plot_disable)
        self.mean.stateChanged.connect(self.plot_disable)
        self.vect.stateChanged.connect(self.plot_disable)
        self.robust.stateChanged.connect(self.plot_disable)

        self.pca.stateChanged.connect(self.plot_disable)
        self.mds.stateChanged.connect(self.plot_disable)
        self.isomap.stateChanged.connect(self.plot_disable)
        self.umap.stateChanged.connect(self.plot_disable)
        self.t_sne.stateChanged.connect(self.plot_disable)


    @Slot()
    def check_disable(self, s):
        """
        Disable load data button if no path is given.
        """
        if not self.file_name.text():
            self.load.setEnabled(False)
        else:
            self.load.setEnabled(True)


    @Slot()
    def plot_disable(self):
        """
        Disable plotting when two or less methods are selected.
        """
        norm_ops, dim_ops = self.plot_options()
        if len(norm_ops) < 2 or len(dim_ops) < 2:
            self.plot.setEnabled(False)
            self.color_plot.setEnabled(False)
        else:
            self.plot.setEnabled(True)
            self.color_plot.setEnabled(True)


    @Slot()
    def load_data(self):
        """
        Load data from given path. File can be excel or csv supported datatypes like .csv, .txt, .data
        """
        try:
            ftype = self.file_name.text().split('.')[-1]
            if ftype == 'xlsx' or ftype == 'xls' or ftype == 'xlsb':
                data = pd.read_excel(self.file_name.text())
            else:
                data = pd.read_csv(self.file_name.text())
            self.data = data
            self.update_table()
        except FileNotFoundError:
            self.handle_error("File not found\nIf file is in the same folder with this code just type the file name\nFor example file.csv")
            

    @Slot()
    def drop_column(self):
        """
        Drop columns according to column index
        """
        try:
            cols = [int(x) for x in self.drop_cols.text().split(',')]
            self.data = self.data.drop(self.data.columns[cols], axis=1)
            self.update_table()
        except (IndexError, ValueError):
            self.handle_error("Invalid index value\nUse index numbers starting from 0\nFor example 0,1,2 or just 0 etc.")


    @Slot()
    def change_index(self):
        """
        Change index column in dataframe and update table.
        """
        try:
            self.data = self.data.set_index(self.index_col.text())
            self.update_table()
        except KeyError:
            self.handle_error("Column not found")
    

    @Slot()
    def plot_data(self):
        """
        Plot data without the need of class information. Plots just datapoints.
        Converts data columns to numeric if needed, updates the table after that.
        Transforms data according to selected options. Calls plot window class to visualize data.
        """
        norm_options, dim_options = self.plot_options()
        self.convert_to_numeric()
        self.update_table()
        transformed_data = transformations.transform_data(self.data, norm_options, dim_options, [self.isomap_neighbors.value(), self.umap_neigbors.value(), self.umap_dist.value(), self.tsne_neighbors.value()])
        self.dialog = PlotWindow(transformed_data, norm_options, dim_options)
        self.dialog.show()


    @Slot()
    def plot_colors(self):
        """
        Plot data with class information. Requires determined index column in dataframe. 
        Converts data columns to numeric if needed, updates the table after that.
        Transforms data according to selected options. Calls plot window class to visualize data.
        """
        norm_options, dim_options = self.plot_options()
        self.convert_to_numeric()
        self.update_table()
        sorted_data, colors, classes = self.sort_data()
        transformed_data = transformations.transform_data(sorted_data, norm_options, dim_options, [self.isomap_neighbors.value(), self.umap_neigbors.value(), self.umap_dist.value(), self.tsne_neighbors.value()])
        self.dialog = PlotWindow(transformed_data, norm_options, dim_options, np.array([colors, classes]), color=True)
        self.dialog.show()


    @Slot()
    def quit_application(self):
        QApplication.quit()

    def plot_options(self):
        """
        Return selected plotting options
        """
        norm_button_states = [self.minmax.isChecked(), self.norm.isChecked(), self.mean.isChecked(), self.vect.isChecked(), self.robust.isChecked()]
        dim_button_states = [self.pca.isChecked(), self.mds.isChecked(), self.isomap.isChecked(), self.umap.isChecked(), self.t_sne.isChecked()]

        norm_methods = ['Minmax', 'Standard scaler', 'Mean', 'Unit vector', 'Robust scaler']
        dim_methods = ['PCA', 'MDS', 'Isomap', 'UMAP', 't-SNE']

        norm_options = []
        dim_options= []

        for i, norm_method in enumerate(norm_methods):
            if norm_button_states[i]:
                norm_options.append(norm_method)
        for i, dim_method in enumerate(dim_methods):
            if dim_button_states[i]:
                dim_options.append(dim_method)

        return norm_options, dim_options


    def handle_error(self, errorMessage):
        """
        Display error window with errorMessage.
        """
        self.dialog = ErrorWindow(errorMessage)
        self.dialog.show()


    def update_table(self):
        """
        Updates main screen table consisting pandas dataframe of data
        """
        head = self.data.head(30)
        headers = list(head)
        self.table.setRowCount(head.shape[0])
        self.table.setColumnCount(head.shape[1])
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setVerticalHeaderLabels(head.index)

        head_array = head.values
        for row in range(head.shape[0]):
            for col in range(head.shape[1]):
                self.table.setItem(row, col, QTableWidgetItem(str(head_array[row,col])))


    def sort_data(self):
        """
        Sort data for color plotting.
        Returns sorted data, matching colors and found classes
        """
        sorted_data = self.data.sort_index(axis=0)
        rows = list(sorted_data.index)
        colors = np.unique(rows, return_inverse=True)[1]
        classes = list(sorted_data.index.unique())
    
        return sorted_data, colors, classes


    def convert_to_numeric(self):
        """
        Drops all NA values, parses object type columns and transfers them into numeric dtype
        """
        self.data.dropna(inplace=True)

        for col in self.data.columns:
            if self.data[col].dtype == np.object:
                p = re.compile(r'[^\d.-]+')
                self.data[col] = self.data[col].str.strip()
                self.data[col] = [p.sub('', x) for x in self.data[col]]

        self.data = self.data.apply(pd.to_numeric, errors='coerce')


class MainWindow(QMainWindow):
    def __init__(self, widget):
        QMainWindow.__init__(self)
        self.setWindowTitle("Data analyzer")

        # Menu
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("File")

        # Exit QAction
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.exit_app)

        self.file_menu.addAction(exit_action)
        self.setCentralWidget(widget)

    @Slot()
    def exit_app(self, checked):
        QApplication.quit()


class ErrorWindow(QMainWindow):
    def __init__(self, errorMessage, parent=None):
        super(ErrorWindow, self).__init__(parent)
        self.setWindowTitle("Error")

        # Display error message
        self.message = QLabel(self)
        self.message.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
        self.message.setText(errorMessage)

        # Push button for closing window
        self.ok = QPushButton("Ok")
        self.ok.setStyleSheet("background-color:#eb8c34;")

        # Page layout
        layout = QVBoxLayout()
        layout.addWidget(self.message)
        layout.addWidget(self.ok)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.ok.clicked.connect(self.close_window)

    @Slot()
    def close_window(self):
        self.close()


class PlotWindow(QMainWindow):
    def __init__(self, data, norm_options, dim_options, color_info=[], color=False, parent=None):
        super(PlotWindow, self).__init__(parent)
        self.setWindowTitle("Plot Window")

        self.norm_options = norm_options
        self.dim_options = dim_options
        self.data = data

        # Variables for color plot
        if color:
            self.colors = color_info[0]
            self.classes = color_info[1]

        # Initialize figure and canvas for plotting with matplotlib
        self.figure = Figure()
        self.axes = self.figure.subplots(nrows=len(self.norm_options), ncols=len(self.dim_options))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Plotting options
        if not color:
            self.regular_plot()
        else:
            self.color_plot()

        # Create page layout with toolbar and canvas
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.show()


    def regular_plot(self):
        """
        Function to plot the datapoints without class information
        """
        for i, row in enumerate(self.axes): 
            for j, col in enumerate(row):
                col.scatter(self.data[i][j][:,0],self.data[i][j][:,1], s=15)
                col.set_title('{} {}'.format(self.norm_options[i], self.dim_options[j]), fontsize=12)
                col.set_yticks([])
                col.set_xticks([])


    def color_plot(self):
        """
        Plot datapoints with class information. Requires color and class values
        """
        for i, row in enumerate(self.axes):
            for j, col in enumerate(row):
                col.scatter(self.data[i][j][:,0],self.data[i][j][:,1], s=15, c=np.squeeze(self.colors), cmap='rainbow')
                col.set_title('{} {}'.format(self.norm_options[i], self.dim_options[j]), fontsize=12)
                col.set_yticks([])
                col.set_xticks([])
                
                if i == (len(self.norm_options)-1) and j == (len(self.dim_options)-1):
                    info = [plt.plot([],[], marker='o', color=cm.rainbow(int(k)/(len(self.classes)-1)), linestyle='None', label=self.classes[k])[0]  for k in range(len(self.classes))]
                    col.legend(handles=info, loc='center left', bbox_to_anchor=(1.01, 0.75), prop={'size':8})


if __name__ == "__main__":
    # Qt Application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    # QWidget
    widget = Widget()
    # QMainWindow using QWidget as central widget
    window = MainWindow(widget)
    window.resize(1000, 800)
    window.show()

    # Execute application
    sys.exit(app.exec_())
