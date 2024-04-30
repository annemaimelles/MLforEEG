# MLforEEG
My final project for Introduction to Problem solving and Programming in Python Spring 2024:
https://youtu.be/vOwQQPZPb2c

I use Emacs Macro to write the code using MobaXterm toolbox.

My work is based on Masoom J. Desai previous work who is assistant professor in department of neurology in University of New Mexico.
In this project I attempted to beat her "Accuracy (Test)" numbers of
normal/abnormal EEG classifier.

Overview of the data
-----------------------------------------------------------------------------------------------

- data was represented as vectors:

Electroencephalography (EEG) data represented as vectors
typically refers to the recording of electrical activity
along the scalp produced by the firing of neurons

(1) Time Series Data:
Each vector represents the voltage changes at a particular electrode over a period of time, creating a time series data set.

(2) Dimensionality:
If you have data from multiple electrodes, each electrode's time series can be thought of as one dimension in a multi-dimensional dataset.
For instance, if you have 32 electrodes, you might have 32 dimensions, where each dimension corresponds to the data from one electrode.

(3) Sampling and Resolution:
The data in each vector is sampled at a certain rate (e.g., 250 Hz), meaning measurements are taken 250 times per second per electrode.
Each sample point is a scalar value representing the voltage at a specific time point.

(4) Vector Representation: Mathematically, each electrode's data can be represented as a vector in an N-dimensional space (N being the number of sample points).
If you concatenate or stack these vectors, you can form a matrix representing all electrodes over a time interval, useful for further analysis like signal processing or machine learning.

-----------------------------------------------------------------------------------------------
Masoom J. Desai work outcomes:
cd data/isip/exp/tuh_eeg/exp_4048

run the original classifier:
src/unm_rf_open.py data/all_train.csv_60 data/all_eval.csv_60

Accuracy (Train):   100.0000%

Confusion Matrix (Train):

[[1306    0]
 [   0 1212]]
 
Accuracy (Test):    67.1642%

Confusion Matrix (Test):

[[106  42]
 [ 46  74]]

The original code uses Random Forest enseble learning
method for classification, regression and other
tasks that operates by constructing a multitude
of decision trees at training time. 

I started out trying out different machine learning algorithms:

Neural Network (unm_rf_open_nn.py):
-----------------------------------------------------------------------------------------------

nedc_130_[1]: src/unm_rf_open_nn.py data/all_train.csv_60 data/all_eval.csv_60

Accuracy (Train):    59.1739%

Confusion Matrix (Train):

[[698 608]
 [420 792]]
 
Accuracy (Test):    51.4925%

Confusion Matrix (Test):

[[74 74]
 [56 64]]


Support Vector Machine (unm_rf_open_svm.py):
-----------------------------------------------------------------------------------------------

nedc_130_[1]: src/unm_rf_open_svm.py data/all_train.csv_60 data/all_eval.csv_60

Accuracy (Train):    55.9968%

Confusion Matrix (Train):

[[1156  150]
 [ 958  254]]
 
Accuracy (Test):    54.1045%

Confusion Matrix (Test):

[[119  29]
 [ 94  26]]

K-Nearest Neighbors(unm_rf_open_knn.py):
-----------------------------------------------------------------------------------------------

nedc_130_[1]: src/unm_rf_open_knn.py data/all_train.csv_60 data/all_eval.csv_60

*k=5

Accuracy (Train):    77.9587%

Confusion Matrix (Train):

[[1069  237]
 [ 318  894]]
 
Accuracy (Test):    60.8209%

Confusion Matrix (Test):

[[94 54]
 [51 69]]

I sticked with KNN model and tested out different k values* to bring accuracy(Train) up to 100%
*k-value defines how many neighbors will be checked to determine the classification of a specific query point

*k= 20
-----------------------------------------------------------------------------------------------

Accuracy (Train):    70.2542%

Confusion Matrix (Train):

[[1051  255]
 [ 494  718]]
 
Accuracy (Test):    63.0597%

Confusion Matrix (Test):

[[106  42]
 [ 57  63]]

*k =10
-----------------------------------------------------------------------------------------------

Accuracy (Train):    73.0739%

Confusion Matrix (Train):

[[1110  196]
 [ 482  730]]
 
Accuracy (Test):    64.1791%

Confusion Matrix (Test):

[[111  37]
 [ 59  61]]

*k=1
-----------------------------------------------------------------------------------------------

Accuracy (Train):   100.0000%

Confusion Matrix (Train):

[[1306    0]
 [   0 1212]]
 
Accuracy (Test):    56.3433%

Confusion Matrix (Test):

[[83 65]
 [52 68]]

k=1 gave me the desired result.

Channel labels that were represented in the data
-----------------------------------------------------------------------------------------------

Now my goal was to get accuracy (Test) as high as possible.

I tested out all the channel labels that were represented in the data. There are 20 channel labels in EEG data:

(FP1, FP2, F7, F8, F3, F4, CZ, FZ, PZ, P2, C3, C4, T3, T4, P3, P4, T5, T6, O1, O2)

Channel label O1 gave me the highest Accuracy(Test) score

nedc_130_[1]: src/unm_rf_open_knn.py data/all_train.csv_60 data/all_eval.csv_60

Accuracy (Train):   100.0000%

Confusion Matrix (Train):

[[1306    0]
 [   0 1212]]
 
Accuracy (Test):    65.2985%

Confusion Matrix (Test):

[[102  46]
 [ 47  73]]
 
1D Convolutional Nerual Network (CNN)
-----------------------------------------------------------------------------------------------
I tried to intertwine 1D Convolutional Neural Network (CNN)

however, it did not work out since it is difficult to make two classifiers work because

defining the variables and making different libaries work is harder. Following error occurs:

nedc_130_[1]: src/unm_rf_open_knn_cnn.py data/all_train.csv_60 data/all_eval.csv_60
2024-04-29 17:00:39.440552: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-04-29 17:00:40.197328: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Traceback (most recent call last):
  File "/data/isip/exp/tuh_eeg/exp_4048/src/unm_rf_open_knn_cnn.py", line 120, in <module>
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
  File "/data/isip/tools/linux_x64/common/anaconda3/conda/envs/nedc/lib/python3.9/site-packages/keras/src/utils/traceback_utils.py", line 122, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/data/isip/tools/linux_x64/common/anaconda3/conda/envs/nedc/lib/python3.9/site-packages/keras/src/ops/operation_utils.py", line 221, in compute_conv_output_shape
    raise ValueError(
ValueError: Computed output size would be negative. Received `inputs shape=(None, 0, 1)`, `kernel shape=(3, 1, 64)`, `dilation_rate=[1]`.

Principal component analysis (PCA) model
-----------------------------------------------------------------------------------------------
I also tried to add PCA model. After changing n_components the data the same 65% accuracy as I got without PCA and only with KNN:

n_components= 0.70
accuracy (Test): 57%

n_components= 10
accuracy (Test): 63%

n_components= 30
Accuracy (Test): 64.9254%

n_components= 100
Accuracy (Test): 65%

What I learned
-----------------------------------------------------------------------------------------------
- how to set up virtual environment
#using venv
python -m venv myprojectenv
source myprojectenv/bin/activate  # On Windows use `myprojectenv\Scripts\activate`
pip install tensorflow numpy

- how to split windows in emacs and move inbetween them:
C-x o (jump from one to another)
C-x 1 (delate a window)
C-x 0 (make a new window)

- adjusting k value
tried different numbers, k=1 proves to be most accurate
- change channel labels classifier reads
tried all channels, for knn most accurate is O1 (occipital area),
meanwhile for randomforest it is T3 (temporal)
conclusion: different classifiers are good at reading different channel labels

- adjust data to certain classifier

Future Plans
-----------------------------------------------------------------------------------------------
- Change the features of the csv files
- Try out multilayer architectures and their applications.

Valuable sources:
-----------------------------------------------------------------------------------------------
Abeer Al-Nafjan. "Feature selection of EEG signals in neuromarketing". National Library of Medicine. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9138093/

Annushree Bablani, Damodar Reddy Edla, Shubham Dodia "Classification of EEG Data using k-Nearest Neighbor approach for Concealed Information Test" Science Direct. https://www.sciencedirect.com/science/article/pii/S1877050918320891

