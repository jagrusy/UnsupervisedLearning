Running the Algorithms

All of the algorithms, related charts, and csv result data can be found at: https://github.com/jagrusy/UnsupervisedLearning

All of the project requirements are in the `requirements.txt` file

`pip install -r requirements.txt`

The code is split up into 5 "experiments" with some experiments having an 'a' and 'b' part.

Running an algorithm (01_experiment.py) file will generate the associated
charts which will be saved in the `Figs` folder named 01_experiment.py.

The experiments can all be run using the following:

    python 01a_clustering.py
    python 01b_clustering_EM_visualization.py
    python 02a_reduction.py
    python 02b_reduction_rand_proj.py
    python 03a_reduction_clustering_kmeans.py
    python 03b_reduction_clustering_EM.py
    python 04_reduction_neural_network.py
    python 05_clustering_neural_network.py

Each algorithm file utilizes the `util.py` file to gather and preprocess the data. In order for the data to be
processed correctly it should be stored in the `Data` folder.

* Note the second dataset is the digits dataset from sci-kit learn which is not included in the `Data` folder.
