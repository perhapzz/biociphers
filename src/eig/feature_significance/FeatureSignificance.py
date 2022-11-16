"""
FeatureSignificance.py
Enhanced Integrated Gradient allows users to assess class-wide feature significance.
We compute significance for different features using one-sided t-test (positive tail) and apply different corrections
for multiple comparisons problem to give a list of significant features with controlled false positives.
Defines:
+ class FeatureSignificance(object)
"""
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats import multitest

MULT_BASELINE = 3
np.random.seed(seed=500)
smallest_float_val = 2.2250738585072014e-308

class FeatureSignificance(object):
    """
    Class for feature significance computation for attributions generated with EIG.
    """

    def __init__(self):
        self.all_features_significance = []
        self.all_features_attributions = []
        self.significant_features = []

    def compute_feature_significance(self, attributions, attributions_null, feature_names, correction_alpha=0.01,
                                     correction_method='bonferroni', sig_threshold=0.05):
        """
        Compute feature significance by comparing attributions
        :param attributions: list/np.array(), Attributions for the class of interest
        :param attributions_null: list/np.array(), attributions for selecting background set
        :param feature_names: np.array(), names of different features
        :param correction_alpha: FWER for correction method
        :param correction_method: str, (default='bonferroni')
                'bonferroni' : one-step correction
                'sidak' : one-step correction
                'holm-sidak' : step down method using Sidak adjustments
                'holm' : step-down method using Bonferroni adjustments
                'simes-hochberg' : step-up method (independent)
                'hommel' : closed method based on Simes tests (non-negative)
                'fdr_bh' : Benjamini/Hochberg (non-negative)
                'fdr_by' : Benjamini/Yekutieli (negative)
                'fdr_tsbh' : two stage fdr correction (non-negative)
                'fdr_tsbky' : two stage fdr correction (non-negative)
        :param sig_threshold: float, Significance threshold (default=0.05)
        :return: np.array() of significant features with feature names and binary indicator
        (significant: 1, not significant: 0)
        """

        # if we are given lists, Get attributions and attributions null in correct shape
        if isinstance(attributions, list) and isinstance(attributions_null, list):
            assert len(attributions) == 1
            attributions = attributions[0]
            null_shape = list(attributions.shape)
            null_shape[0] = 0
            attributions_null_concatenate = np.empty(tuple(null_shape))
            for ii in attributions_null:
                attributions_null_concatenate = np.concatenate((attributions_null_concatenate, ii))
            attributions_null = np.array(attributions_null_concatenate)

        # If more than two dimension to the features, flatten them.
        if len(attributions.shape) > 2 and len(attributions_null.shape) > 2 and len(feature_names.shape) > 2:
            attributions_flattened = np.array([np.array(ii).flatten() for ii in attributions])
            attributions_null_flattened = np.array([np.array(ii).flatten() for ii in attributions_null])
            feature_names_flatten = np.array([np.array(ii).flatten() for ii in feature_names])
            print(attributions_flattened.shape, attributions_null_flattened.shape, feature_names_flatten.shape)

        # If only attributions and attributions_null need to be flattened.
        elif len(feature_names.shape) == 2:
            attributions_flattened = np.array([np.array(ii).flatten() for ii in attributions])
            attributions_null_flattened = np.array([np.array(ii).flatten() for ii in attributions_null])
            feature_names_flatten = feature_names.flatten()
            print(attributions_flattened.shape, attributions_null_flattened.shape, feature_names_flatten.shape)

        # Nothing needs to be flattened.
        else:
            attributions_flattened = attributions
            attributions_null_flattened = attributions_null
            feature_names_flatten = feature_names
            print(attributions_flattened.shape, attributions_null_flattened.shape, feature_names_flatten.shape)

        # Compute pvalues
        pvalues = []
        valid_pvalues = []
        invalid_pvalues = []
        features_attributions = []
        for ii in range(attributions_flattened.shape[1]):
            attr_null_rand = np.random.choice(np.array(attributions_null_flattened[:, ii]),
                                              size=attributions_flattened.shape[0], replace=False)
            features_attributions.append(np.median(attributions_flattened[:, ii]))
            stats, pvalue = ttest_ind(np.abs(attributions_flattened[:, ii]),
                                      np.abs(attr_null_rand), equal_var=False)
            pval_greater = (pvalue if stats > 0 else 1 - pvalue) / 2
            # Handle nan pvalues separately
            if np.isnan(pval_greater):
                invalid_pvalues.append(ii)
            else:
                avoid_zero = max(pval_greater, smallest_float_val)
                pvalues.append(avoid_zero)
                valid_pvalues.append(ii)

        pvalues = np.array(pvalues)
        rejected, pvalue_corrected, alpha1, alpha2 = multitest.multipletests(pvalues,
                                                                             alpha=correction_alpha,
                                                                             method=correction_method)
        # Make all nan pvalues 1
        valid_pvalues = np.array(valid_pvalues)
        invalid_pvalues = np.array(invalid_pvalues)
        pvalue_all = np.zeros((attributions_flattened.shape[1]))
        pvalue_all[valid_pvalues] = pvalue_corrected
        # pvalue_all[valid_pvalues] = pvalues
        if len(invalid_pvalues) >= 1:
            pvalue_all[invalid_pvalues] = 1.0

        # flatten feature names
        feature_names_flatten = feature_names_flatten.reshape((len(feature_names_flatten), 1))

        pvalue_corrected = pvalue_all.reshape(len(pvalue_all), 1)

        features_attributions = np.array(features_attributions)

        features_attributions = features_attributions.reshape(len(features_attributions), 1)

        # Save feature attributions
        self.all_features_attributions = np.concatenate((feature_names_flatten, features_attributions), axis=1)

        # Save feature significance
        self.all_features_significance = np.concatenate((feature_names_flatten, pvalue_corrected), axis=1)

        idx = np.where(np.array(self.all_features_significance[:, 1], dtype=float) <= sig_threshold)[0]

        # Find significant features
        self.significant_features = np.array(self.all_features_significance, copy=True)

        self.significant_features[:, 1] = 0
        self.significant_features[idx, 1] = 1
        print("number of significant features: ", len(idx))
        return self.significant_features

