class GMMFitParamHolder:
    """Container class to provide the hyperparameters pertaining to all GMM fits.
    See mi.py for the meaning of each parameter.
    """
    def __init__(self, init_type="random_sklearn", scale=None, 
                 threshold_fit=1e-5, reg_covar=1e-15, max_iter=10000):
        self.init_type = init_type
        self.scale = scale
        self.threshold_fit = threshold_fit
        self.reg_covar = reg_covar
        self.max_iter = max_iter

class SelectComponentsParamHolder:
    """Container class to provide the hyperparameters pertaining to cross-validation
    and selecting the number of GMM components. See mi.py for the meaning of each parameter.
    """
    def __init__(self, n_inits=3, n_folds=2, metric_method='valid',
                 threshold_components=1e-5, patience=1, max_components=50,
                 fixed_components_number=0):
        self.n_inits = n_inits
        self.n_folds = n_folds
        assert metric_method == 'valid' or metric_method == 'aic' or metric_method == 'bic', f"metric_method must be either 'valid', 'aic' or 'bic, found '{metric_method}'"
        self.metric_method = metric_method
        assert threshold_components >= 0, f"`threshold_components` must be a non-negative number, found '{threshold_components}'"
        self.threshold_components = threshold_components
        assert patience >= 1, f"patience should be at least 1, found {patience}."
        self.patience = patience
        self.max_components = max_components
        self.fixed_components_number = fixed_components_number
        self.fixed_components = True if self.fixed_components_number > 0 else False

class MIDistParamHolder:
    """Container class to provide the hyperparameters pertaining to the MI distribution.
    See mi.py for the meaning of each parameter.
    """
    def __init__(self, n_bootstrap=50, integral_method='MC', 
                MC_samples=1e5):
        self.n_bootstrap = n_bootstrap
        assert integral_method == 'MC' or integral_method == 'quad', f"`integral_method` must be a either 'MC' or 'quad', found '{integral_method}'"
        self.integral_method = integral_method
        self.MC_samples = MC_samples

        