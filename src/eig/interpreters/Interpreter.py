"""
Interpreter.py

Abstract (unimplemented) class for getting attributions from different paths for LIG
Defines:
+ class Interpreter(object)

"""


class Interpreter(object):
    """
    This is the parent class of all interpreters.

    """
    def attributions(self):
        """ Unimplemented function to obtain attributions for features in data.

        Returns
        -------
        np.ndarray(shape=(num_samples, num_features))
            feature attributions for all samples

        Notes
        -----
        This is not implemented for Interpreter. Subclasses should
        keep the same function signature
        """
        err_msg = "This is an abstract class; use its subclasses."
        raise NotImplementedError(err_msg)

    def attributions_subgroups(self, subgroups):
        """ Unimplemented function to obtain attributions for features subgroups in data.

        Parameters
        ----------
        subgroups: np.ndarray(shape=(num_features, 2))
            two columns array where the first column specifies feature index and the second
            column indicates the feature subgroup

        Returns
        -------
        np.ndarray(shape=(num_samples, num_features_subgroups))
            feature subgroup attributions for all samples

        Notes
        -----
        This is not implemented for Interpreter. Subclasses should
        keep the same function signature
        """
        err_msg = "This is an abstract class; use its subclasses."
        raise NotImplementedError(err_msg)
