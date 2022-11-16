import argparse
import tensorflow as tf
import numpy as np
from eig.interpreters.DeepNetworkModel import DeepNetworkModel
from eig.interpreters.LinearInterpreter import LinearInterpreter
from eig.feature_significance.FeatureSignificance import FeatureSignificance
from data_prep import read_ge_data_interpret


def get_pc_pvalues(cancer_attributes, normal_attributes, feature_names, prefix, selected_gene_class):
    """
    Get ranked list of feature attributions with FDR corrected p-values
    :param cancer_attributes: np.array, array containing attributions for cancer class
    :param normal_attributes: np.array, array containing attributions for normal class
    :param feature_names: np.array, gene names
    :param prefix: str, path where output will be stored
    :param selected_gene_class: str, protein_coding_genes or lncRNA_genes
    :return: None
    """

    # Find attributions for cancer class when compared to a null distribution with normal and cancer class
    target_attributes = [cancer_attributes]
    null_attributes = [cancer_attributes, normal_attributes]

    print("target_attributes: ", len(cancer_attributes))
    print("null_attributes: ", len(cancer_attributes) + len(normal_attributes))

    # Initialize the feature significance class from EIG
    fs = FeatureSignificance()

    # Get significant features
    significant_features = fs.compute_feature_significance(target_attributes,
                                                           null_attributes,
                                                           feature_names,
                                                           correction_alpha=0.01,
                                                           correction_method='fdr_bh')

    # Sort by attribution values
    idx_sort = np.argsort(np.array(fs.all_features_significance[:, 1], dtype=float))
    assert np.array_equal(fs.all_features_significance[:, 0],
                          fs.all_features_attributions[:, 0]), "array names not equal"

    # Write gene name, attribution values and p-values to a file.
    feature_all = np.concatenate((fs.all_features_significance[idx_sort, 0:1],
                                  fs.all_features_attributions[idx_sort, 1:2],
                                  fs.all_features_significance[idx_sort, 1:2]), axis=1)

    header_val = "Gene-Name\tMedian-Attributions\tCorrected P-values (FDR <= 1%)"
    np.savetxt(prefix + selected_gene_class + "_interpret_pvalues_abs.txt", feature_all, header=header_val, fmt="%s\t%s\t%s")
    print("pc: ", len(np.where(np.array(significant_features[:, 1], dtype=int) == 1)[0]))

    print("all_features_attributions: ", fs.all_features_attributions)
    print("all_features_significance: ", fs.all_features_significance)
    print("feature_all: ", feature_all)


def get_ge_attributions(data, dnn_model):
    """
    Get attributions for cancer and normal samples using protein coding or lncRNA genes.
    :param data: dict, data dict with normal and cancer samples and labels.
    :param dnn_model: str, path to dnn tensorflow model
    :return: list, attributions for normal and cancer class and gene names
    """
    # Get normal and cancer indices
    normal_idx = 0
    cancer_idx = 1
    #normal_data_idx = np.where(data['labels'][4800:5000, normal_idx] == 1)[0]
    #cancer_data_idx = np.where(data['labels'][4800:5000, cancer_idx] == 1)[0]

    normal_data_idx = np.where(data['labels'][:, normal_idx] == 1)[0]
    cancer_data_idx = np.where(data['labels'][:, cancer_idx] == 1)[0]

    print("normal_data_idx: ", len(normal_data_idx))
    print("cancer_data_idx: ", len(cancer_data_idx))

    # Load the tensorflow model
    binary_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, name="binary_loss")
    loss_val = [binary_loss]

    discriminator = tf.keras.models.load_model(dnn_model,
                                               custom_objects={'Loss': loss_val},
                                               compile=False)
    discriminator.compile(loss=loss_val, optimizer=tf.keras.optimizers.Adam())

    discriminator.summary()

    # Initialize DeepNetworkModel to run EIG
    batch_size = 500
    deep_model = DeepNetworkModel(discriminator, cancer_idx, batch_size)

    # Select the baseline type
    baseline_type = "median"  # can be replaced with: "k-means", "close", "random"

    # In the case of cancer attributions, normal samples are the baseline and cancer samples are the samples
    baseline_data = data["data"][normal_data_idx, :]
    sample_data_all = data["data"][cancer_data_idx, :]

    # This can be increased based on GPU memory capacity
    skip_factor = 20
    rem = len(cancer_data_idx) % skip_factor
    mod_val = len(cancer_data_idx) + rem

    min_val_cancer = np.arange(0, mod_val, skip_factor)
    max_val_cancer = np.arange(skip_factor, mod_val + skip_factor, skip_factor)
    print(min_val_cancer)
    print(max_val_cancer)

    # This is the main loop to get attributions for cancer class, this can take long (few hours) without a GPU.
    all_attrs_cancer = np.empty((0, sample_data_all.shape[1]))
    for i, j in zip(min_val_cancer, max_val_cancer):
        if i > len(cancer_data_idx):
            break
        if j > len(cancer_data_idx):
            j = len(cancer_data_idx)
            #break
        print(i, j, len(cancer_data_idx), len(sample_data_all))
        sample_data = sample_data_all[i:j]
        baseline_tuple = [baseline_data]
        sample_tuple = [sample_data]

        li = LinearInterpreter(baseline_tuple,
                               sample_tuple,
                               nn_model=deep_model,
                               baseline_type=baseline_type, no_points=250)
        attributes_li = li.attributions()
        print("attributes_li: ", attributes_li.shape)

        all_attrs_cancer = np.concatenate((all_attrs_cancer, attributes_li), axis=0)
        print("all_attrs_cancer: ", all_attrs_cancer.shape)

    # This can be increased based on GPU memory capacity
    batch_size = 500
    deep_model = DeepNetworkModel(discriminator, normal_idx, batch_size)
    
    # In the case of cancer attributions, normal samples are the baseline and cancer samples are the samples
    baseline_data = data["data"][cancer_data_idx, :]
    sample_data_all = data["data"][normal_data_idx, :]

    skip_factor = 20
    rem = len(normal_data_idx) % skip_factor
    mod_val = len(normal_data_idx) + rem

    min_val_normal = np.arange(0, mod_val, skip_factor)
    max_val_normal = np.arange(skip_factor, mod_val + skip_factor, skip_factor)
    print(min_val_normal)
    print(max_val_normal)

    # This is the main loop to get attributions for normal class, this can take long (few hours) without a GPU.
    all_attrs_normal = np.empty((0, sample_data_all.shape[1]))
    for i, j in zip(min_val_normal, max_val_normal):
        if i > len(normal_data_idx):
            break
        if j > len(normal_data_idx):
            j = len(normal_data_idx)
            #break
        print(i, j, len(normal_data_idx))
        sample_data = sample_data_all[i:j]
        baseline_tuple = [baseline_data]
        sample_tuple = [sample_data]

        li = LinearInterpreter(baseline_tuple,
                               sample_tuple,
                               nn_model=deep_model,
                               baseline_type=baseline_type)
        attributes_li = li.attributions()
        print("attributes_li: ", attributes_li.shape)
        all_attrs_normal = np.concatenate((all_attrs_normal, attributes_li), axis=0)
        print("all_attrs_normal: ", all_attrs_normal.shape)

    return all_attrs_normal, all_attrs_cancer, data["subset_genes"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_data_file',
        type=str,
        help="File to read data from"
    )

    parser.add_argument(
        '--dnn_model',
        type=str,
        help="save outputs to this path"
    )

    parser.add_argument(
        '--selected_gene_class',
        type=str,
        default='protein_coding_genes',
        choices=['all', 'rbp_genes', 'tf_genes', 'res_rbp_genes', 'rbp_tf_genes', 'lncRNA_genes',
                 'protein_coding_genes'],
        help="selected feature"
    )

    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help="output path"
    )

    args = parser.parse_args()

    # Get normalized data
    data = read_ge_data_interpret(input_data_file=args.input_data_file,
                                  selected_gene_class=args.selected_gene_class)
    # Get gene expression attributions
    all_attrs_normal, all_attrs_cancer, subset_genes = get_ge_attributions(data, args.dnn_model)

    # Get p-values and store attributions
    get_pc_pvalues(all_attrs_cancer, all_attrs_normal, subset_genes, args.output_path, args.selected_gene_class)
