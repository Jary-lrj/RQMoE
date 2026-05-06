import numpy as np


def get_mean_and_std(arr: np.array):
    print(f"{arr.mean():.4f}±{arr.std():.4f}")


def get_descend_svd(arr: np.array):
    normed = np.round(arr / arr.max(), 4)
    print(', '.join(map(str, normed)))


# hot
get_descend_svd(np.array([26.042599, 2.8548489, 2.1051397, 1.965772, 1.8518757, 1.7629663,
                          1.7091876, 1.6659317, 1.6529068, 1.5873816]))
# mid
get_descend_svd(np.array([23.659609, 1.9544414, 1.8073481, 1.7753912, 1.6842028, 1.4598328,
                          1.4298471, 1.3912297, 1.3604232, 1.2984847]))
# cold
get_descend_svd(np.array([14.657656, 1.1267265, 1.0962055, 1.051613, 0.9788847, 0.87383604,
                          0.8499262, 0.8259749, 0.8090454, 0.78788894]))
# ultra-cold
get_descend_svd(np.array([10.034386, 0.7542566, 0.7369708, 0.68655854, 0.6391703, 0.5771263,
                          0.5613382, 0.5499047, 0.53753245, 0.5175254]))
