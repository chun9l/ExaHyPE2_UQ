import torch
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler

# Detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Use double precision for stability
torch.set_default_dtype(torch.float64)

# random seeds
np.random.seed(7)
torch.manual_seed(5)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(5)

class WeightedGP:
    """
    A Gaussian Process model with per-data-point noise weights,
    supporting RBF, Matérn 3/2, and Matérn 5/2 kernels.

    Inputs (X) and outputs (y) are automatically standardised via
    sklearn StandardScaler before training. Predictions are returned
    in the original output scale.
    """

    def __init__(self, x_train, y_train, kernel='rbf', device=None):
        self.kernel = kernel
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.log_lengthscale = None
        self.log_variance = None
        self.log_noise_weights = None
        self.nll_history = []
        
        self.x_train = x_train
        self.y_train = y_train

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    # ------------------------------------------------------------------
    # Kernels
    # ------------------------------------------------------------------

    def _rbf_kernel(self, X1, X2, lengthscales, amplitude):
        X1 = X1 / lengthscales
        X2 = X2 / lengthscales
        pairwise_dists = torch.cdist(X1, X2, p=2)
        return amplitude * torch.exp(-0.5 * pairwise_dists ** 2)

    def _matern_kernel(self, X1, X2, lengthscales, amplitude, nu=1.5):
        X1 = X1 / lengthscales
        X2 = X2 / lengthscales
        d = torch.cdist(X1, X2, p=2)
        if nu == 1.5:
            return (1 + d) * amplitude * torch.exp(-d)
        elif nu == 2.5:
            return (1 + d + (d ** 2) / 3) * amplitude * torch.exp(-d)
        else:
            raise ValueError("Only Matérn ν=1.5 and ν=2.5 are supported.")

    def _compute_kernel(self, X1, X2, lengthscales, amplitude):
        if self.kernel == 'rbf':
            return self._rbf_kernel(X1, X2, lengthscales, amplitude)
        elif self.kernel == 'matern52':
            return self._matern_kernel(X1, X2, lengthscales, amplitude, nu=2.5)
        elif self.kernel == 'matern32':
            return self._matern_kernel(X1, X2, lengthscales, amplitude, nu=1.5)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    # ------------------------------------------------------------------
    # Scaling helpers
    # ------------------------------------------------------------------

    def _scale_inputs(self, x_np):
        """Fit (if not already fitted) and transform X; returns a tensor."""
        return torch.tensor(
            self.x_scaler.transform(x_np), dtype=torch.float64, device=self.device
        )

    def _scale_outputs(self, y_np):
        """Fit (if not already fitted) and transform y; returns a tensor."""
        return torch.tensor(
            self.y_scaler.transform(y_np), dtype=torch.float64, device=self.device
        )

    def _unscale_mean(self, mean_tensor):
        """Inverse-transform a predicted mean back to the original output scale."""
        mean_np = mean_tensor.detach().cpu().numpy()
        return torch.tensor(
            self.y_scaler.inverse_transform(mean_np),
            dtype=torch.float64, device=self.device
        )

    def _unscale_cov(self, cov_tensor):
        """
        Scale a predicted covariance matrix back to the original output scale.
        Cov_original = Cov_scaled * y_std^2
        """
        y_std = torch.tensor(self.y_scaler.scale_, dtype=torch.float64, device=self.device)
        return cov_tensor * (y_std ** 2)

    # ------------------------------------------------------------------
    # Negative log marginal likelihood
    # ------------------------------------------------------------------

    def _nll(self, x, y, lengthscale, variance, noise_weights):
        K = self._compute_kernel(x, x, lengthscale, variance)
        K = K + torch.diag(noise_weights)

        L = torch.linalg.cholesky(K)

        alpha = torch.cholesky_solve(y, L)
        log_det_K = 2.0 * torch.sum(torch.log(torch.diagonal(L)))

        N = x.shape[0]
        nll = (
            0.5 * y.T.mm(alpha)
            + 0.5 * log_det_K
            + 0.5 * N * torch.log(torch.tensor(2.0 * torch.pi))
        )
        return nll.squeeze()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        n_steps=1000,
        lr=0.01,
        learn_weights=True,
        bounds=None,
    ):
        """
        Standardise inputs/outputs, then optimise kernel hyperparameters
        (and optionally per-point noise weights) by maximising the log
        marginal likelihood.

        Parameters
        ----------
        x_train      : (N, D) array-like or tensor
        y_train      : (N, 1) array-like or tensor
        n_steps      : number of Adam optimisation steps
        lr           : Adam learning rate
        learn_weights: True  – learn per-point weights from scratch
                       False – fix all weights to -inf (zero noise)
                       Tensor – initialise weights from this tensor
        bounds       : optional list of (min, max) clamp bounds per param
        """
        # Convert to numpy for sklearn, then fit scalers
        x_np = self.x_train.detach().cpu().numpy() if torch.is_tensor(x_train) else np.array(x_train)
        y_np = self.y_train.detach().cpu().numpy() if torch.is_tensor(y_train) else np.array(y_train)

        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)

        self.x_scaler.fit(x_np)
        self.y_scaler.fit(y_np)

        x_scaled = self._scale_inputs(x_np)
        y_scaled = self._scale_outputs(y_np)

        # Store scaled training data for use in predict()
        self.x_train_scaled = x_scaled
        self.y_train_scaled = y_scaled

        self.log_lengthscale = torch.nn.Parameter(
            torch.zeros(x_scaled.shape[1], device=self.device) - 1.0
        )
        self.log_variance = torch.nn.Parameter(
            torch.tensor(0.0, device=self.device)
        )

        params = [self.log_lengthscale, self.log_variance]

        if isinstance(learn_weights, bool):
            if learn_weights:
                self.log_noise_weights = torch.nn.Parameter(
                    torch.zeros(x_scaled.shape[0], device=self.device) - 6.0
                )
                params.append(self.log_noise_weights)
            else:
                self.log_noise_weights = (
                    torch.zeros(x_scaled.shape[0], device=self.device) - torch.inf
                )
        else:
            self.log_noise_weights = torch.nn.Parameter(
                torch.log(learn_weights)
            )
            params.append(self.log_noise_weights)

        optimizer = optim.Adam(params, lr=lr)
        self.nll_history = []

        for step in range(n_steps):
            optimizer.zero_grad()

            noise_weights = torch.exp(self.log_noise_weights)
            lengthscale   = torch.exp(self.log_lengthscale)
            variance      = torch.exp(self.log_variance)

            loss = self._nll(x_scaled, y_scaled, lengthscale, variance, noise_weights)
            loss += torch.exp(self.log_noise_weights).sum()   # L1 penalty

            loss.backward()
            optimizer.step()

            if bounds is not None:
                with torch.no_grad():
                    for i, p in enumerate(params):
                        p.clamp_(bounds[i])

            self.nll_history.append(loss.item())

            if (step + 1) % 100 == 0 or step == 0:
                print(
                    f"Step {step + 1}/{n_steps}, NLL = {loss.item():.4f}, "
                    f"lengthscale = {[f'{v.item():.4f}' for v in lengthscale]}, "
                    f"variance = {variance.item():.4f}, "
                    f"log_noise (mean) = {self.log_noise_weights.mean().item():.4f}"
                )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, x_test, return_scaled=False):
        """
        Return the posterior (mean, covariance) at x_test in the
        *original* output scale (unless return_scaled=True).

        Parameters
        ----------
        x_test        : (M, D) array-like or tensor
        return_scaled : if True, return mean/cov in the standardised scale

        Returns
        -------
        pred_mean : (M, 1) tensor  – in original output units
        pred_cov  : (M, M) tensor  – in original output units²
        """
        if self.log_lengthscale is None:
            raise RuntimeError("Call fit() before predict().")

        x_np = x_test.detach().cpu().numpy() if torch.is_tensor(x_test) else np.array(x_test)
        x_scaled = self._scale_inputs(x_np)

        lengthscale   = torch.exp(self.log_lengthscale)
        variance      = torch.exp(self.log_variance)
        noise_weights = torch.exp(self.log_noise_weights)

        K_xx = self._compute_kernel(self.x_train_scaled, self.x_train_scaled, lengthscale, variance)
        K_xs = self._compute_kernel(x_scaled,            self.x_train_scaled, lengthscale, variance)
        K_ss = self._compute_kernel(x_scaled,            x_scaled,            lengthscale, variance)

        K_xx = K_xx + torch.diag(noise_weights)

        L     = torch.linalg.cholesky(K_xx)
        alpha = torch.cholesky_solve(self.y_train_scaled, L)

        pred_mean_scaled = K_xs @ alpha
        v                = torch.linalg.solve(L, K_xs.T)
        pred_cov_scaled  = K_ss - v.T.mm(v)

        if return_scaled:
            return pred_mean_scaled, pred_cov_scaled

        pred_mean = self._unscale_mean(pred_mean_scaled)
        pred_cov  = self._unscale_cov(pred_cov_scaled)

        return pred_mean, pred_cov

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lengthscale(self):
        return torch.exp(self.log_lengthscale).detach()

    @property
    def variance(self):
        return torch.exp(self.log_variance).detach()

    @property
    def noise_weights(self):
        return torch.exp(self.log_noise_weights).detach()

