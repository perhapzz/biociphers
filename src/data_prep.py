import os
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, scale


def seed_everything(seed):
    """
    Seed all random libraries to get deterministic behavior
    :param seed: int, value to set up random seeds
    :return: None
    """
    print("current seed is", seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'


def read_input_data_ge(input_data_file, selected_gene_class):
    """
    Read protein-coding or lncRNA genes for normal and cancer samples and return normalized train and test data
    :param input_data_file: str, npz object containing protein-coding or lncRNA data for normal versus cancer
    :param selected_gene_class: str, whether to process protein-coding or lncRNA genes
        for normal versus cancer
    :return: normalized data divided into train and three test datasets and size of the feature set
    """
    seed_everything(seed=42)
    print("load input data for {}....".format(selected_gene_class))
    data_dict = np.load(input_data_file, encoding='latin1')

    # Select protein coding genes or lncRNA
    if selected_gene_class == "protein_coding_genes":
        x_data = data_dict["x_normalized_pc"]
        unseen_data = data_dict["unseen_x_normalized_pc"]
    else:
        x_data = data_dict["x_normalized_lncrna"]
        unseen_data = data_dict["unseen_x_normalized_lncrna"]


    y_labels = data_dict["all_labels"]
    y_labels_binary = []
    for ii in y_labels:
        if 'tumor' in ii:
            y_labels_binary.append('cancer')
        else:
            y_labels_binary.append('normal')

    y_labels_binary = np.array(y_labels_binary, dtype=str)
    y_labels_binary_one_hot = np.zeros((len(y_labels_binary), 2))
    idx_val_normal = np.where(y_labels_binary == 'normal')[0]
    idx_val_cancer = np.where(y_labels_binary == 'cancer')[0]
    y_labels_binary_one_hot[idx_val_normal, 0] = 1
    y_labels_binary_one_hot[idx_val_cancer, 1] = 1

    # Get combined labels for datasets and normal versus cancer
    y_datasets = data_dict["all_datasets"]
    combined_y_labels = []
    for ii, jj in zip(y_labels_binary, y_datasets):
        val = ii + "_" + jj
        combined_y_labels.append(val)

    combined_y_labels = np.array(combined_y_labels, dtype=str)

    combined_y_labels_all = np.array(combined_y_labels, copy=True)
    x_data_all = np.array(x_data, copy=True)
    y_labels_binary_one_hot_all = np.array(y_labels_binary_one_hot, copy=True)
    y_datasets_all = np.array(y_datasets, copy=True)

    # Remove independent test set with healthy lung and cancer samples
    unseen_test_label = 'PRJEB2784'
    unseen_index = np.where(data_dict["all_datasets"] == unseen_test_label)[0]

    # Remove independent test set from whole data after batch correction
    seen_index = np.setdiff1d(np.arange(0, len(data_dict["all_datasets"]), 1), unseen_index)
    combined_y_labels = combined_y_labels_all[seen_index]
    x_data = x_data_all[seen_index]
    y_labels_binary_one_hot = y_labels_binary_one_hot_all[seen_index]
    y_datasets = y_datasets_all[seen_index]

    # Make separate objects for independent test sets
    combined_y_labels_test = combined_y_labels_all[unseen_index]
    x_data_test = x_data_all[unseen_index]
    y_labels_binary_one_hot_test = y_labels_binary_one_hot_all[unseen_index]
    y_datasets_test = y_datasets_all[unseen_index]

    # Get test sets from same tissue types and datasets as the training set.
    skf = StratifiedKFold(n_splits=5)
    data_set = dict()
    data_set["unseen_test_data"] = x_data_test
    data_set["unseen_test_labels"] = y_labels_binary_one_hot_test
    data_set["unseen_test_datasets"] = y_datasets_test
    data_set["unseen_test_label_names"] = combined_y_labels_test
    data_set["unseen_data"] = unseen_data
    data_set["unseen_labels"] = data_dict["unseen_y_labels_binary_one_hot"]
    data_set["unseen_datasets"] = data_dict["unseen_all_datasets"]
    data_set["unseen_label_names"] = data_dict["unseen_all_labels"]

    for cv_index, test_index in skf.split(x_data, combined_y_labels):
        data_set["cv_label_names"] = combined_y_labels[cv_index]
        data_set["cv_data"] = x_data[cv_index, :]
        data_set["test_data"] = x_data[test_index, :]
        data_set["test_label_names"] = combined_y_labels[test_index]

        data_set["cv_labels"] = y_labels_binary_one_hot[cv_index, :]
        data_set["test_labels"] = y_labels_binary_one_hot[test_index, :]
        data_set["cv_datasets"] = y_datasets[cv_index]
        data_set["test_datasets"] = y_datasets[test_index]

        scaler = StandardScaler().fit(data_set["cv_data"])
        data_set["cv_data"] = scaler.transform(data_set["cv_data"])
        data_set["test_data"] = scaler.transform(data_set["test_data"])
        data_set["unseen_test_data"] = scaler.transform(data_set["unseen_test_data"])
        data_set["unseen_data"] = scaler.transform(data_set["unseen_data"])
        print("data loaded for {}".format(selected_gene_class))
        return data_set, data_set["cv_data"].shape[1]


def read_input_data_splicing(input_data_file):
    """
    Return normalized splicing data divided in train and test set.
    :param input_data_file:
    :return:
    """
    seed_everything(seed=42)
    print("Loading splicing data....")
    data_dict = np.load(input_data_file, encoding='latin1')

    x_data = data_dict["x_normalized"]
    y_labels = data_dict["all_labels"]

    y_labels_binary = []
    for ii in y_labels:
        if 'tumor' in ii:
            y_labels_binary.append('cancer')
        else:
            y_labels_binary.append('normal')

    y_labels_binary = np.array(y_labels_binary, dtype=str)
    y_labels_binary_one_hot = np.zeros((len(y_labels_binary), 2))
    idx_val_normal = np.where(y_labels_binary == 'normal')[0]
    idx_val_cancer = np.where(y_labels_binary == 'cancer')[0]
    y_labels_binary_one_hot[idx_val_normal, 0] = 1
    y_labels_binary_one_hot[idx_val_cancer, 1] = 1

    # Get combined labels for datasets and normal versus cancer
    y_datasets = data_dict["all_datasets"]
    combined_y_labels = []
    for ii, jj in zip(y_labels_binary, y_datasets):
        val = ii + "_" + jj
        combined_y_labels.append(val)

    combined_y_labels = np.array(combined_y_labels, dtype=str)

    combined_y_labels_all = np.array(combined_y_labels, copy=True)
    x_data_all = np.array(x_data, copy=True)
    y_labels_binary_one_hot_all = np.array(y_labels_binary_one_hot, copy=True)
    y_datasets_all = np.array(y_datasets, copy=True)

    # Remove independent test set with healthy lung and cancer samples
    unseen_test_label = 'PRJEB2784'
    unseen_index = np.where(data_dict["all_datasets"] == unseen_test_label)[0]

    # Remove independent test set from whole data after batch correction
    seen_index = np.setdiff1d(np.arange(0, len(data_dict["all_datasets"]), 1), unseen_index)
    combined_y_labels = combined_y_labels_all[seen_index]
    x_data = x_data_all[seen_index]
    y_labels_binary_one_hot = y_labels_binary_one_hot_all[seen_index]
    y_datasets = y_datasets_all[seen_index]

    # Make separate objects for independent test sets
    combined_y_labels_test = combined_y_labels_all[unseen_index]
    x_data_test = x_data_all[unseen_index]
    y_labels_binary_one_hot_test = y_labels_binary_one_hot_all[unseen_index]
    y_datasets_test = y_datasets_all[unseen_index]

    # Get test sets from same tissue types and datasets as the training set.
    skf = StratifiedKFold(n_splits=5)
    data_set = dict()
    data_set["unseen_test_data"] = x_data_test
    data_set["unseen_test_labels"] = y_labels_binary_one_hot_test
    data_set["unseen_test_datasets"] = y_datasets_test
    data_set["unseen_test_label_names"] = combined_y_labels_test


    for cv_index, test_index in skf.split(x_data, combined_y_labels):
        data_set["cv_label_names"] = combined_y_labels[cv_index]
        data_set["cv_data"] = x_data[cv_index, :]
        data_set["test_data"] = x_data[test_index, :]
        data_set["test_label_names"] = combined_y_labels[test_index]

        data_set["cv_labels"] = y_labels_binary_one_hot[cv_index, :]
        data_set["test_labels"] = y_labels_binary_one_hot[test_index, :]
        data_set["cv_datasets"] = y_datasets[cv_index]
        data_set["test_datasets"] = y_datasets[test_index]

        scaler = StandardScaler().fit(data_set["cv_data"])
        data_set["cv_data"] = scaler.transform(data_set["cv_data"])
        data_set["test_data"] = scaler.transform(data_set["test_data"])
        data_set["unseen_test_data"] = scaler.transform(data_set["unseen_test_data"])
        print("Loaded splicing data.")
        return data_set, data_set["cv_data"].shape[1]


def read_ge_data_interpret(input_data_file, selected_gene_class):
    """
    Read data for downstream interpretation.
    :param input_data_file: str, path to file with protein coding genes and lncrna data
    :param selected_gene_class: str, protein_coding_genes or lncRNA_genes
    :return: dict, normalized data and labels
    """
    seed_everything(seed=42)
    print("Loading {} data for interpretation....".format(selected_gene_class))
    data_dict = np.load(input_data_file, encoding='latin1')

    if selected_gene_class == "protein_coding_genes":
        x_data = data_dict["pc_norm_interpret"]
        subset_genes = data_dict["pc_gene_names"]
        scaler = StandardScaler().fit(x_data)
        x_data = scaler.transform(x_data)
    else:
        x_data = data_dict["lncrna_norm_interpret"]
        subset_genes = data_dict["lncRNA_gene_names"]
        scaler = StandardScaler().fit(x_data)
        x_data = scaler.transform(x_data)
    print("x_data", np.min(x_data), np.max(x_data), np.mean(x_data), np.std(x_data))
    y_labels = data_dict["all_labels"]
    y_labels_binary = []
    for ii in y_labels:
        if 'tumor' in ii:
            y_labels_binary.append('cancer')
        else:
            y_labels_binary.append('normal')

    y_labels_binary = np.array(y_labels_binary, dtype=str)
    y_labels_binary_one_hot = np.zeros((len(y_labels_binary), 2))
    idx_val_normal = np.where(y_labels_binary == 'normal')[0]
    idx_val_cancer = np.where(y_labels_binary == 'cancer')[0]
    y_labels_binary_one_hot[idx_val_normal, 0] = 1
    y_labels_binary_one_hot[idx_val_cancer, 1] = 1

    y_datasets = data_dict["all_datasets"]

    combined_y_labels = []
    for ii, jj in zip(y_labels_binary, y_datasets):
        val = ii + "_" + jj
        combined_y_labels.append(val)

    combined_y_labels = np.array(combined_y_labels, dtype=str)
    data_set = dict()
    data_set["sample_names"] = data_dict["sample_names"]
    data_set["label_names"] = combined_y_labels
    data_set["data"] = x_data
    data_set["labels"] = y_labels_binary_one_hot
    data_set["datasets"] = y_datasets
    data_set["subset_genes"] = subset_genes
    data_set["all_labels"] = data_dict["all_labels"]
    data_set["all_datasets"] = data_dict["all_datasets"]
    print("Loaded {} data for interpretation.".format(selected_gene_class))
    return data_set

def read_splicing_data_interpret(input_data_file):
    """
    Read splicing data for downstream interpretations
    :param input_data_file: str, path to npz
    :return:
    """
    seed_everything(seed=42)
    print("Loading splicing data for interpretation....")
    data_dict = np.load(input_data_file, encoding='latin1')

    x_data = data_dict["x_normalized"]

    y_labels = data_dict["all_labels"]
    y_labels_binary = []
    for ii in y_labels:
        if 'tumor' in ii:
            y_labels_binary.append('cancer')
        else:
            y_labels_binary.append('normal')

    y_labels_binary = np.array(y_labels_binary, dtype=str)
    y_labels_binary_one_hot = np.zeros((len(y_labels_binary), 2))
    idx_val_normal = np.where(y_labels_binary == 'normal')[0]
    idx_val_cancer = np.where(y_labels_binary == 'cancer')[0]
    y_labels_binary_one_hot[idx_val_normal, 0] = 1
    y_labels_binary_one_hot[idx_val_cancer, 1] = 1

    y_datasets = data_dict["all_datasets"]

    combined_y_labels = []
    for ii, jj in zip(y_labels_binary, y_datasets):
        val = ii + "_" + jj
        combined_y_labels.append(val)

    combined_y_labels = np.array(combined_y_labels, dtype=str)
    data_set = dict()
    data_set["sample_names"] = data_dict["sample_names"]
    data_set["label_names"] = combined_y_labels
    data_set["data"] = x_data
    data_set["labels"] = y_labels_binary_one_hot
    data_set["datasets"] = y_datasets
    data_set["lsv_names"] = data_dict["lsv_names"]
    data_set["all_labels"] = data_dict["all_labels"]
    data_set["all_datasets"] = data_dict["all_datasets"]
    print("Loaded splicing data for interpretation.")
    return data_set
