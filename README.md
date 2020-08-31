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

Note! Umap requires that you have 64 bit windows and 64 bit python version to work with parallel option.
If your system doesn't meet the requirements, you can go to umap folder in the virtual environment and modify python files where it says 'parallel = True' to
'parallel = False'

## Examples

![Main window](https://github.com/markmult/Data-visualizer/tree/master/Examples/visualizer.png?raw=true)


![Transformations and visualization with above options](https://github.com/markmult/Data-visualizer/tree/master/Examples/visualizer%20output.png?raw=true)
