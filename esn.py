# esn.py
import numpy as np

class EchoStateNetwork:
    """
    Echo State Network (Reservoir Computing)
    - Fixed random Win, W
    - Train only Wout using ridge regression
    """

    def __init__(
        self,
        input_dim: int,
        reservoir_size: int = 300,
        spectral_radius: float = 0.9,
        sparsity: float = 0.1,
        leak_rate: float = 0.3,
        ridge_alpha: float = 1e-3,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.N = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak = leak_rate
        self.alpha = ridge_alpha
        self.rng = np.random.default_rng(seed)

        # Input weights (N x (input_dim+1)) including bias
        self.Win = self.rng.uniform(-1, 1, size=(self.N, self.input_dim + 1)).astype(np.float32)

        # Reservoir weights (N x N)
        W = self.rng.uniform(-1, 1, size=(self.N, self.N)).astype(np.float32)

        # Sparsify
        mask = self.rng.random((self.N, self.N)) < self.sparsity
        W *= mask.astype(np.float32)

        # Spectral radius scaling
        eigvals = np.linalg.eigvals(W)
        rho = np.max(np.abs(eigvals)) if eigvals.size else 1.0
        if rho == 0:
            rho = 1.0
        W = (W / rho) * self.spectral_radius

        self.W = W.astype(np.float32)

        # Readout (trained later): output_dim x (N+1)
        self.Wout = None

        # State
        self.x = np.zeros((self.N,), dtype=np.float32)

    def reset_state(self):
        self.x[:] = 0.0

    def _step(self, u: np.ndarray):
        """
        u: (input_dim,) float
        """
        u = u.astype(np.float32)
        u_bias = np.concatenate([np.array([1.0], dtype=np.float32), u])  # (input_dim+1,)

        pre = self.Win @ u_bias + self.W @ self.x
        x_new = np.tanh(pre)

        # leaky integrator
        self.x = (1.0 - self.leak) * self.x + self.leak * x_new
        return self.x

    def collect_states(self, U: np.ndarray, washout: int = 20):
        """
        U: (T, input_dim)
        returns X: (T-washout, N+1) with bias appended
        """
        self.reset_state()
        states = []
        for t in range(U.shape[0]):
            self._step(U[t])
            if t >= washout:
                states.append(np.concatenate([np.array([1.0], dtype=np.float32), self.x.copy()]))
        return np.vstack(states) if states else np.zeros((0, self.N + 1), dtype=np.float32)

    def fit(self, U: np.ndarray, Y: np.ndarray, washout: int = 20):
        """
        Ridge regression:
        Wout = Y^T X (X^T X + alpha I)^-1
        U: (T, input_dim)
        Y: (T-washout, output_dim) or (T, output_dim) -> we align after washout
        """
        X = self.collect_states(U, washout=washout)  # (T-washout, N+1)

        if Y.shape[0] == U.shape[0]:
            Yt = Y[washout:]
        else:
            Yt = Y

        # Ridge
        XtX = X.T @ X
        I = np.eye(XtX.shape[0], dtype=np.float32)
        inv = np.linalg.inv(XtX + self.alpha * I)
        self.Wout = (Yt.T @ X) @ inv  # (output_dim, N+1)
        return self

    def predict_proba(self, u: np.ndarray):
        """
        For binary output: returns p(congestion) using sigmoid.
        """
        if self.Wout is None:
            raise RuntimeError("ESN not trained. Train or load Wout first.")

        self._step(u)
        x_bias = np.concatenate([np.array([1.0], dtype=np.float32), self.x])
        y = (self.Wout @ x_bias).reshape(-1)  # raw
        # sigmoid
        p = 1.0 / (1.0 + np.exp(-y))
        return p

    def save(self, path: str):
        np.savez(
            path,
            Win=self.Win,
            W=self.W,
            Wout=self.Wout,
            input_dim=self.input_dim,
            N=self.N,
            spectral_radius=self.spectral_radius,
            sparsity=self.sparsity,
            leak=self.leak,
            alpha=self.alpha,
        )

    @staticmethod
    def load(path: str):
        d = np.load(path, allow_pickle=True)
        esn = EchoStateNetwork(
            input_dim=int(d["input_dim"]),
            reservoir_size=int(d["N"]),
            spectral_radius=float(d["spectral_radius"]),
            sparsity=float(d["sparsity"]),
            leak_rate=float(d["leak"]),
            ridge_alpha=float(d["alpha"]),
        )
        esn.Win = d["Win"]
        esn.W = d["W"]
        esn.Wout = d["Wout"]
        esn.reset_state()
        return esn