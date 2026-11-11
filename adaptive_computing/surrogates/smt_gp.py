from smt.surrogate_models import KRG
from smt.applications.mfk import MFK
from smt.applications.mixed_integer import MixedIntegerSurrogateModel, MixedIntegerKrigingModel
from adaptive_computing.surrogates.base import SurrogateModelBase
import numpy as np

class SMT_GP(SurrogateModelBase):
    """
    A wrapper class for using Surrogate Modeling Toolbox (SMT)
     Kriging or Multi-fidelity Kriging as surrogate model.
    
    Attributes:
        surrogate_model (list): List of surrogate models for each fidelity level.
        
    Methods:
        __init__(dataset, smt_kwargs=None): Initializes the SMT_GP.
        train(x_data, y_data): Trains the surrogate models.
        predict_values(x_data, fidelity_level=-1): Predicts values using the surrogate model.
        predict_variances(x_data, fidelity_level=-1): Predicts variances using the surrogate model.
    """
    
    def __init__(self, dataset, smt_kwargs=None, design_space=None, i_output=0, mfk_kwargs=None):
        """
        Initializes the SMT_GP with the dataset and optional SMT-specific keyword arguments.
        
        Args:
            dataset (DatasetBase): The dataset to use for training and prediction.
            smt_kwargs (dict, optional): Additional keyword arguments for configuring SMT models. Defaults to None.
            design_space (DesignSpace, optional): SMT design space for mixed-type problems. Defaults to None.
            i_output (int, optional): Index of output dimension to model (for multi-output datasets). Defaults to 0.
                SMT Kriging models only support single outputs, so this parameter specifies which
                output dimension from a multi-output dataset should be used.
            mfk_kwargs (dict, optional): Extra keyword arguments for the multi-fidelity (MFK) levels
                only, merged over smt_kwargs. Use it for MFK-only options that the level-0 KRG
                rejects, e.g. {"rho_regr": "linear"}. While a level has too few points to fit a
                non-constant rho regression, that level is trained with rho_regr="constant".
        """
        super().__init__(dataset)
        
        # Validate output index
        if i_output < 0 or i_output >= dataset.n_out:
            raise ValueError(f"i_output={i_output} is invalid. Dataset has {dataset.n_out} outputs (valid range: 0 to {dataset.n_out-1})")
        
        self.i_output = i_output
        if dataset.n_out > 1:
            print(f"Note: SMT GP modeling output dimension {i_output} of {dataset.n_out} total outputs")

        if smt_kwargs is None:
            smt_kwargs = {}
        mfk_kwargs = {**smt_kwargs, **(mfk_kwargs or {})}
        self.rho_used = [None] * self.n_fidelity  # rho_regr actually used at the last train
        self._n_in = dataset.n_in
        self._rho_requested = {}
            
        # Get design space from dataset if available
        if design_space is None and hasattr(dataset, 'design_space'):
            design_space = dataset.design_space
        
        self.surrogate_model = []
        for i_fidelity in range(self.n_fidelity):
            if self.mixed_type:
                # Use regular KRG - mixed types are handled in the optimization algorithm
                self.surrogate_model.append(KRG(**smt_kwargs, print_global=False))
                print("Note: Mixed-type optimization using KRG surrogate with discrete enumeration in acquisition optimization")
            elif self.multifidelity and i_fidelity > 0:
                self.surrogate_model.append(MFK(
                    **mfk_kwargs,
                    print_global=False))
            else:
                self.surrogate_model.append(KRG(
                    **smt_kwargs,
                    print_global=False))
        
        self.untrained = True # used to track if the surrogate model has never been trained

    def _train_impl(self, x_data, y_data):
        """
        Internal SMT_GP training implementation.
        This method receives only validated, unmasked training data.
        
        Args:
            x_data (list): List of input data arrays for each fidelity level (unmasked only).
            y_data (list): List of output data arrays for each fidelity level (unmasked only).
        """
        for i_fidelity in range(self.n_fidelity):
            # Extract only the selected output dimension for GP modeling
            y_selected = y_data[i_fidelity][:, self.i_output:self.i_output+1]  # Keep as 2D array
            
            for ii_fidelity in range(i_fidelity):
                y_ii_selected = y_data[ii_fidelity][:, self.i_output:self.i_output+1]
                self.surrogate_model[i_fidelity].set_training_values(x_data[ii_fidelity], y_ii_selected, name=ii_fidelity)
            self.surrogate_model[i_fidelity].set_training_values(x_data[i_fidelity], y_selected)
            self._train_level(i_fidelity, x_data[i_fidelity].shape[0])
        
        self.untrained = False

    _N_REGR_TERMS = {"constant": lambda d: 1, "linear": lambda d: d + 1,
                     "quadratic": lambda d: (d + 1) * (d + 2) // 2}

    def _train_level(self, i_fidelity, n_points):
        """Train one level. An MFK level whose rho regression has more terms than its
        points can support (n_points <= rho terms + delta trend terms), or that fails to
        train with it, is trained with a constant rho instead. SMT reads rho_regr again at
        prediction time, so the fallback stays in the options until the next train, which
        retries the requested rho."""
        model = self.surrogate_model[i_fidelity]
        if not isinstance(model, MFK):
            model.train()
            return
        requested = self._rho_requested.setdefault(i_fidelity, model.options["rho_regr"])
        model.options["rho_regr"] = requested
        if requested != "constant":
            n_terms = self._N_REGR_TERMS[requested](self._n_in) + self._N_REGR_TERMS[model.options["poly"]](self._n_in)
            if n_points > n_terms:
                try:
                    model.train()
                    self.rho_used[i_fidelity] = requested
                    return
                except (ValueError, np.linalg.LinAlgError) as exc:
                    print(f"Note: MFK level {i_fidelity} failed to train with rho_regr={requested!r} ({exc}); "
                          "using 'constant' until the next train")
            model.options["rho_regr"] = "constant"
        model.train()
        self.rho_used[i_fidelity] = "constant"

    def predict_values(self, x_data, fidelity_level=-1):
        """
        Predicts values using the surrogate model at a specified fidelity level.
        
        Args:
            x_data (N samples, N input dimension): Input data.
            fidelity_level (int): Fidelity level to use. Defaults to -1 (highest fidelity).
        
        Returns:
            np.ndarray: Predicted values.
        """
        if self.untrained:
            raise Exception("Attempting to evaluate the surrogate model values, but user never called train() since initializing the surrogate.")
        return self.surrogate_model[fidelity_level].predict_values(np.atleast_2d(x_data))

    def predict_variances(self, x_data, fidelity_level=-1):
        """
        Predicts variances using the surrogate model at a specified fidelity level.
        
        Args:
            x_data (N samples, N input dimension): Input data.
            fidelity_level (int): Fidelity level to use. Defaults to -1 (highest fidelity).
        
        Returns:
            np.ndarray: Predicted variances.
        """
        if self.untrained:
            raise Exception("Attempting to evaluate the surrogate model variances, but user never called train() since initializing the surrogate.")
        return self.surrogate_model[fidelity_level].predict_variances(np.atleast_2d(x_data))


class ConstrainedSMT_GP(SMT_GP):
    """
    A wrapper class for using Constrained Surrogate Modeling Toolbox (SMT) models as surrogate models.
    
    Attributes:
        constraint_func (function): Function to compute constraints.
        
    Methods:
        __init__(dataset, constraint_func, smt_kwargs=None, i_out=0): Initializes the ConstrainedSMT_GP.
        predict_constraint(x_data, fidelity_level=-1): Predicts constraints using the constraint function.
    """
    
    def __init__(self, dataset, constraint_func, smt_kwargs=None, i_out=0):
        """
        Initializes the ConstrainedSMT_GP with the dataset, constraint function, optional SMT-specific keyword arguments, and output index.
        
        Args:
            dataset (DatasetBase): The dataset to use for training and prediction.
            constraint_func (function): Function to compute constraints based on input data.
            smt_kwargs (dict, optional): Additional keyword arguments for configuring SMT models. Defaults to None.
            i_out (int, optional): Index of the output for constrained optimization. Defaults to 0.
                This parameter is passed as i_output to the parent SMT_GP class.
        """
        super().__init__(dataset, smt_kwargs=smt_kwargs, i_output=i_out)
        self.constraint_func = constraint_func
        
    def predict_constraint(self, x_data, fidelity_level=-1):
        """
        Predicts constraints using the constraint function at a specified fidelity level.
        
        Args:
            x_data (N samples, N input dimension): Input data.
            fidelity_level (int): Fidelity level to use. Defaults to -1 (highest fidelity).
        
        Returns:
            float: Predicted constraint value.
        """
        return self.constraint_func(x_data)
