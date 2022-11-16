import argparse
import tensorflow as tf
import numpy as np
from sklearn import metrics
from data_prep import read_input_data_ge
from joblib import load

# DNN for predicting healthy versus tumor state
class AETissueAndCancer:
    # Define the network
    def __init__(self, dnn_model):
        """
        Initialize the class with the DNN network
        :param dnn_model: str, previously trained DNN model
        """
        binary_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, name="binary_loss")
        loss_val = [binary_loss]
        self._discriminator = tf.keras.models.load_model(dnn_model,
                                                         custom_objects={'Loss': loss_val}, compile=False)
        self._discriminator.compile(loss=loss_val, optimizer=tf.keras.optimizers.Adam())

        print("Discriminator net")
        self._discriminator.summary()

    # test models
    def test(self, data_dict, test_results):
        """
        Evaluate accuracy and AUPRC for three test sets in the pan-cancer manuscript.
        :param data_dict: dict, test data and labels for the DNN model
        :param test_results: dict, data dictionary for saving test results
        :return: None
        """
        test_pred = self._discriminator.predict(data_dict["test_data"])
        y_true = np.argmax(data_dict["test_labels"], axis=1)
        y_scores = test_pred[:, 1]
        y_scores_binary = np.argmax(test_pred, axis=1)

        aupr = metrics.average_precision_score(y_true, y_scores, pos_label=1)
        fscoreOpt = metrics.f1_score(y_true, y_scores_binary, pos_label=1)
        acc = metrics.accuracy_score(y_true, y_scores_binary)

        print("Performance on test set: ")
        print("Fscore: {}".format(fscoreOpt * 100))
        print("Area under PR curve: {}".format(aupr * 100))
        print("accuracy: {}".format(acc * 100))

        unseen_test_pred = self._discriminator.predict(data_dict["unseen_test_data"])
        y_true = np.argmax(data_dict["unseen_test_labels"], axis=1)
        y_scores = unseen_test_pred[:, 1]
        y_scores_binary = np.argmax(unseen_test_pred, axis=1)

        aupr = metrics.average_precision_score(y_true, y_scores, pos_label=1)
        fscoreOpt = metrics.f1_score(y_true, y_scores_binary, pos_label=1)
        acc = metrics.accuracy_score(y_true, y_scores_binary)

        print("Performance on independent test set (PRJEB2784): ")
        print("Fscore: {}".format(fscoreOpt * 100))
        print("Area under PR curve: {}".format(aupr * 100))
        print("accuracy: {}".format(acc * 100))

        unseen_pred = self._discriminator.predict(data_dict["unseen_data"])
        y_true = np.argmax(data_dict["unseen_labels"], axis=1)
        y_scores = unseen_pred[:, 1]
        y_scores_binary = np.argmax(unseen_pred, axis=1)

        aupr = metrics.average_precision_score(y_true, y_scores, pos_label=1)
        fscoreOpt = metrics.f1_score(y_true, y_scores_binary, pos_label=1)
        acc = metrics.accuracy_score(y_true, y_scores_binary)

        print("Performance on blood tissues and tumors: ")
        print("Fscore: {}".format(fscoreOpt * 100))
        print("Area under PR curve: {}".format(aupr * 100))
        print("accuracy: {}".format(acc * 100))

        test_results["test_labels"] = data_dict["test_labels"]
        test_results["test_datasets"] = data_dict["test_datasets"]
        test_results["test_pred"] = test_pred
        test_results["test_label_names"] = data_dict["test_label_names"]

        test_results["unseen_test_labels"] = data_dict["unseen_test_labels"]
        test_results["unseen_test_datasets"] = data_dict["unseen_test_datasets"]
        test_results["unseen_test_pred"] = unseen_test_pred
        test_results["unseen_test_label_names"] = data_dict["unseen_test_label_names"]

        test_results["unseen_labels"] = data_dict["unseen_labels"]
        test_results["unseen_datasets"] = data_dict["unseen_datasets"]
        test_results["unseen_pred"] = unseen_pred
        test_results["unseen_label_names"] = data_dict["unseen_label_names"]

        return test_results


def test_svm(svm_model, data, test_results):
    """
    Print test metrics for the SVM model
    :param svm_model:str, path to the trained SVM model
    :param data: dict, data dictionary with the gene expression data
    :param test_results: dict, data dictionary for saving test results
    :return: None
    """
    print("svm model")
    svm_classifier = load(svm_model)
    test_pred =svm_classifier.predict(data["test_data"])
    y_true = np.argmax(data["test_labels"], axis=1)
    y_scores = test_pred #[:, 1]
    y_scores_binary = test_pred

    aupr = metrics.average_precision_score(y_true, y_scores, pos_label=1)
    fscoreOpt = metrics.f1_score(y_true, y_scores_binary, pos_label=1)
    acc = metrics.accuracy_score(y_true, y_scores_binary)

    print("Performance on test set: ")
    print("Fscore: {}".format(fscoreOpt * 100))
    print("Area under PR curve: {}".format(aupr * 100))
    print("accuracy: {}".format(acc * 100))

    unseen_test_pred = svm_classifier.predict(data["unseen_test_data"])
    y_true = np.argmax(data["unseen_test_labels"], axis=1)
    y_scores = unseen_test_pred
    y_scores_binary = unseen_test_pred

    aupr = metrics.average_precision_score(y_true, y_scores, pos_label=1)
    fscoreOpt = metrics.f1_score(y_true, y_scores_binary, pos_label=1)
    acc = metrics.accuracy_score(y_true, y_scores_binary)

    print("Performance on independent test set (PRJEB2784): ")
    print("Fscore: {}".format(fscoreOpt * 100))
    print("Area under PR curve: {}".format(aupr * 100))
    print("accuracy: {}".format(acc * 100))

    unseen_pred = svm_classifier.predict(data["unseen_data"])
    y_true = np.argmax(data["unseen_labels"], axis=1)
    y_scores = unseen_pred
    y_scores_binary = unseen_pred

    aupr = metrics.average_precision_score(y_true, y_scores, pos_label=1)
    fscoreOpt = metrics.f1_score(y_true, y_scores_binary, pos_label=1)
    acc = metrics.accuracy_score(y_true, y_scores_binary)

    print("Performance on blood tissues and tumors: ")
    print("Fscore: {}".format(fscoreOpt * 100))
    print("Area under PR curve: {}".format(aupr * 100))
    print("accuracy: {}".format(acc * 100))

    test_results["test_pred_svm"] = test_pred
    test_results["unseen_test_pred_svm"] = unseen_test_pred
    test_results["unseen_pred_svm"] = unseen_pred
    return test_results


def test_rf(rf_model, data, test_results):
    """
    Print test metrics for the random forest model
    :param rf_model:str, path to trained random forest model
    :param data: dict, data dictionary with the gene expression data
    :param test_results: dict, data dictionary for saving test results
    :return: None
    """
    print("rf model")
    rf_classifier = load(rf_model)
    test_pred =rf_classifier.predict(data["test_data"])
    y_true = np.argmax(data["test_labels"], axis=1)
    y_scores = test_pred
    y_scores_binary = test_pred

    aupr = metrics.average_precision_score(y_true, y_scores, pos_label=1)
    fscoreOpt = metrics.f1_score(y_true, y_scores_binary, pos_label=1)
    acc = metrics.accuracy_score(y_true, y_scores_binary)

    print("Performance on test set: ")
    print("Fscore: {}".format(fscoreOpt * 100))
    print("Area under PR curve: {}".format(aupr * 100))
    print("accuracy: {}".format(acc * 100))

    unseen_test_pred = rf_classifier.predict(data["unseen_test_data"])
    y_true = np.argmax(data["unseen_test_labels"], axis=1)
    y_scores = unseen_test_pred
    y_scores_binary = unseen_test_pred

    aupr = metrics.average_precision_score(y_true, y_scores, pos_label=1)
    fscoreOpt = metrics.f1_score(y_true, y_scores_binary, pos_label=1)
    acc = metrics.accuracy_score(y_true, y_scores_binary)

    print("Performance on independent test set (PRJEB2784): ")
    print("Fscore: {}".format(fscoreOpt * 100))
    print("Area under PR curve: {}".format(aupr * 100))
    print("accuracy: {}".format(acc * 100))

    unseen_pred = rf_classifier.predict(data["unseen_data"])
    y_true = np.argmax(data["unseen_labels"], axis=1)
    y_scores = unseen_pred
    y_scores_binary = unseen_pred

    aupr = metrics.average_precision_score(y_true, y_scores, pos_label=1)
    fscoreOpt = metrics.f1_score(y_true, y_scores_binary, pos_label=1)
    acc = metrics.accuracy_score(y_true, y_scores_binary)

    print("Performance on blood tissues and tumors: ")
    print("Fscore: {}".format(fscoreOpt * 100))
    print("Area under PR curve: {}".format(aupr * 100))
    print("accuracy: {}".format(acc * 100))

    test_results["test_pred_rf"] = test_pred
    test_results["unseen_test_pred_rf"] = unseen_test_pred
    test_results["unseen_pred_rf"] = unseen_pred
    return test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_data_file',
        type=str,
        help="Input data for testing protein-coding and lncrna genes DNN models"
    )

    parser.add_argument(
        '--selected_gene_class',
        type=str,
        default='protein_coding_genes',
        choices=['lncRNA_genes', 'protein_coding_genes'],
        help="selected feature class"
    )

    parser.add_argument(
        '--dnn_model',
        type=str,
        help="DNN model"
    )
    
    parser.add_argument(
        '--svm_model',
        type=str,
        help="DNN model"
    )
    
    parser.add_argument(
        '--rf_model',
        type=str,
        help="DNN model"
    )

    parser.add_argument(
        '--output_file',
        type=str,
        help="DNN model"
    )

    args = parser.parse_args()
    data, feature_shape = read_input_data_ge(input_data_file=args.input_data_file,
                                          selected_gene_class=args.selected_gene_class
                                          )

    ae_tissue_cancer = AETissueAndCancer(args.dnn_model)

    test_results = dict()
    test_results = ae_tissue_cancer.test(data, test_results)
    test_results = test_svm(args.svm_model, data, test_results)
    test_results = test_rf(args.rf_model, data, test_results)
    np.savez(args.output_file, **test_results)