import numpy as np
from sklearn.model_selection import KFold
import warnings
from gmm_mi.single_fit import single_fit
from gmm_mi.initializations import initialize_parameters

class CrossValidation:
    """Perform cross-validation (CV) to select the best GMM initialization parameters, 
    and thus avoid local minima in the density estimation.

    Parameters
    ----------
    n_components : int
        Number of GMM components currently being fitted.
    n_folds : int, default=2
        Number of folds.
    n_inits : int, default=3
        Number of initializations.
    init_type : {'random', 'minmax', 'kmeans', 'randomized_kmeans', 'random_sklearn', 'kmeans_sklearn'}, default='random_sklearn'
        The method used to initialize the weights, the means, the covariances and the precisions in each fit.
        See utils.initializations for more details.
    cale : float, default=None
        The scale to use to initialize the GMM parameters. Only used if 'init_type' is 'random', 
        'minmax' or 'randomized_sklearn'.
    reg_covar : float, default=1e-15
        The constant term added to the diagonal of the covariance matrices to avoid singularities.
    threshold_fit : float, default=1e-5
        The log-likelihood threshold on each GMM fit used to choose when to stop training.
        Smaller values will improve the fit quality and reduce the chances of stopping at a local optimum,
        while making the code considerably slower. This is equivalent to `tol` in sklearn GMMs.       
    max_iter : int, default=10000
        The maximum number of iterations in each GMM fit. We aim to stop only based on `threshold_fit`, 
        so it is set to a high value. A warning is raised if this threshold is reached; in that case, 
        simply increase this value, or check that you really need such a small `threshold_fit` value.
    """
    def __init__(self, n_components, n_folds=2, n_inits=3, init_type='random_sklearn', 
                 scale=None, reg_covar=1e-15, threshold_fit=1e-5, max_iter=10000):
        self.n_components = n_components
        self.n_folds = n_folds
        self.n_inits = n_inits
        self.init_type = init_type
        self.scale = scale
        self.reg_covar = reg_covar
        self.threshold_fit = threshold_fit
        self.max_iter = max_iter
        
    def _create_containers(self):
        """Create all arrays and lists to store CV results"""
        # create empty arrays to store validation log-likelihoods
        self.val_scores = np.zeros((self.n_inits, self.n_folds))    
        # create empty arrays for final GMM parameters, and loss curves
        self.all_ws = np.zeros((self.n_inits, self.n_folds, self.n_components))
        self.all_ms = np.zeros((self.n_inits, self.n_folds, self.n_components, self.features))
        self.all_ps = np.zeros((self.n_inits, self.n_folds, self.n_components, self.features, self.features))    
        self.all_lcurves = []
        self.convergence_flags = []
        # random split seed is fixed here, but results should be independent of the exact split
        self.kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
    def _run_cross_validation(self, X):        
        """
        Actually run the cross-validation (CV) procedure, filling all arrays and lists with results.
        
        Parameters
        ----------
            X : array-like of shape (n_samples, n_features)
                Data on which CV is performed.
        """
        for random_state in range(self.n_inits):
            # initialize with different seed random_state
            w_init, m_init, c_init, p_init = initialize_parameters(X, random_state=random_state, 
                                                                   n_components=self.n_components, 
                                                                   init_type=self.init_type,
                                                                   scale=self.scale)       
            # perform k-fold CV
            for k_id, (train_indices, valid_indices) in enumerate(self.kf.split(X)):
                X_training = X[train_indices]
                X_validation = X[valid_indices]            
                gmm = single_fit(X=X_training, n_components=self.n_components, reg_covar=self.reg_covar, 
                                 threshold_fit=self.threshold_fit, max_iter=self.max_iter, random_state=random_state, 
                                 w_init=w_init, m_init=m_init, p_init=p_init, val_set=X_validation)
                # we take the mean logL per sample, since folds might have slightly different sizes
                val_score = gmm.score_samples(X_validation).mean()            
                # save current scores, as well as GMM parameters
                self.val_scores[random_state, k_id] = np.copy(val_score)
                self.all_ws[random_state, k_id] = np.copy(gmm.weights_)
                self.all_ms[random_state, k_id] = np.copy(gmm.means_)
                self.all_ps[random_state, k_id] = np.copy(gmm.precisions_)
                # save the loss functions as well
                self.all_lcurves.append(np.copy(gmm.train_loss))
                self.all_lcurves.append(np.copy(gmm.val_loss))
                # save a flag to warn about non-convergence
                convergence_flag = False if gmm.n_iter_ <= 2 else True
                self.convergence_flags.append(convergence_flag)
                
    def fit(self, X):
        """Perform CV on the given data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data on which CV is performed.
        
        Returns
        -------
        results_dict : dict
            A dictionary containing all results from the CV procedure:
            'best_seed': the initialization seed that led to the highest mean log-likelihood across folds.
            'best_val_score': the corresponding value of the highest mean log-likelihood.
            'best_fold_in_init':  the index of the best fold within the best initialization.
            'all_lcurves': list of list with the training and validation curves.
            'all_ws': all weights of the fitted GMM models.
            'all_ms': all means of the fitted GMM models.
            'all_ps': all precisions of the fitted GMM models.
        """
        self.features = X.shape[1]
        self._create_containers()
        self._run_cross_validation(X)
        # select seed with highest val score across the different inits
        avg_val_scores = np.mean(self.val_scores, axis=1)
        best_seed = np.argmax(avg_val_scores)
        best_val_score = np.max(avg_val_scores)
        # within the best fold, also select the model with the highest validation logL
        best_fold_in_init = np.argmax(self.val_scores[best_seed])    
        results_dict = {'best_seed': best_seed, 'best_val_score': best_val_score, 
                        'best_fold_in_init': best_fold_in_init, 'all_lcurves': self.all_lcurves, 
                        'all_ws': self.all_ws, 'all_ms': self.all_ms, 'all_ps': self.all_ps,
                        'convergence_flags': self.convergence_flags}
        return results_dict  