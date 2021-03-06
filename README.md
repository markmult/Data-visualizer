# Data-visualizer
Data visualizer is data analysis tool that allows user to easily do some basic data manipulation and visualize data with various preprocessing methods.

Data visualization is important step in exploratory data analysis when you want to find out structures and patterns in data.  For example it is interesting
to see if data is distributed in different clusters. This information is also required to help selecting methods for machine learning.

With data visualizer you can easily apply common scaling, normalization an dimensionality reduction methods to your data.
User can also manipulate the input data by dropping some of the columns. Program takes care of missing values and parses all unwanted characters from
input columns in a way that columns can be treated as numerical values and selected transformations can be calculated.

User can choose from five different scaling and normalization methods: Minmax scaling, Standard scaler, Mean scaling, Normalizer (unit vector normalization)
and Robust scaler. For dimensionality reduction options are PCA, MDS, Isomap, UMAP and t-SNE. Most of these functions are provided by scikit-learn library.
UMAP uses own umap-learn library. In addition to choosing from these methods, user can tune some of the most common parameters used by these dimensionality
reduction algorithms.

For visualizations user can choose all these methods and compare how outputs differ with different methods. If input data has some kind of class information
it is possible to plot datapoints also with colors based on the class labels. Requirement is that user has set index column before plotting.

**This tool is primarily intended for educational purposes.**

## Setup

I recommend to install and create a virtual enviroment and after that install the required extensions. To do this, you must have python 3.x and python3-pip installed
(The Python installers for Windows include pip) then run following commands:

```
py -m pip install --user virtualenv
py -m venv env
source env\Scripts\activate
```
Next clone this repository or download the files.
```
git clone https://github.com/markmult/Data-visualizer.git
pip install -r relative\path\requirements.txt
```
After you have installed required packages, run main.py

**Note!** Umap requires that you have 64 bit windows and 64 bit python version to work with parallel option.
If your system doesn't meet the requirements, you can go to umap folder in the virtual environment and modify python files where it says 'parallel = True' to
'parallel = False'

**Note!** Program might take some time to start at the first time. If it doesn't seem to start after short waiting, press ctrl+c and try again.

## How to use

1. Load data. If you cloned this repository, one example dataset is accessed with typing data/breast_cancer.csv and then pressing the load data -button
2. Inspect the data table and try to get some idea of the data columns
3. If there seems to be columns that doesn't contain useful information or columns that can't be parsed into numeric datatype, drop them by using index value
4. Set index column if the data has some class information and you want to plot with corresponding colors. Index column is chosen by typing its name
5. Choose scaling, normalization and dimensional reduction methods
6. Tune parameters or use the default ones
7. Plot with or without colors and class information
8. Enjoy the visualizations
9. If you have chosen many different methods, you can press the configure subplots -icon at the top of the plot window and choose Tight layout -option to avoid overlaps

**Note!** It might take some time to calculate the transformations depending on the chosen methods and your computer. Visualizations appear after that.

## Examples

#### Main window with loaded data.
<img src="https://github.com/markmult/Data-visualizer/blob/master/Examples/visualizer.png" width="700">
<br/>

#### Data visualization with above options.
<img src="https://github.com/markmult/Data-visualizer/blob/master/Examples/visualizer%20output.png" width="700">
