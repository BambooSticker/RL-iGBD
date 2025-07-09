import numpy as np

class PPOPolicy:
    def __init__(self, model, env):
        self.model = model
        self.env = env
    def predict(self, obs_norm, deterministic=True):
        action, state = self.model.predict(obs_norm, deterministic=deterministic)
        raw_idx = int(action)
        raw = -1.0 + 2.0 * raw_idx / float(11 - 1)
        mp_gap_tol_u = 0.3
        mp_gap_tol_u_l = 1e-3
        print('raw index:', raw)

        scale = np.sqrt(self.env.obs_rms.var + self.env.epsilon)
        obs_true = obs_norm * scale + self.env.obs_rms.mean
        true_gap = obs_true[0][9]
        print('true gap:', true_gap)

        factor = mp_gap_tol_u_l + (raw + 1.0) * 0.5 * (true_gap - mp_gap_tol_u_l)
        # Clip to original bounds
        factor = [np.clip(factor, mp_gap_tol_u_l, mp_gap_tol_u)]
        print('tolerance', factor)

        return factor, state
    
class OptimalPolicy:
    def __init__(self, env, const=0):
        self.const = const
        self.env = env
    def predict(self, obs_norm, deterministic=True):
        state = 0
        raw_idx = int(self.const)
        raw = -1.0 + 2.0 * raw_idx / float(11 - 1)
        mp_gap_tol_u = 0.3
        mp_gap_tol_u_l = 1e-3
        print('raw index:', raw)

        scale = np.sqrt(self.env.obs_rms.var + self.env.epsilon)
        obs_true = obs_norm * scale + self.env.obs_rms.mean
        true_gap = obs_true[0][9]
        print('true gap:', true_gap)

        factor = mp_gap_tol_u_l + (raw + 1.0) * 0.5 * (true_gap - mp_gap_tol_u_l)
        # Clip to original bounds
        factor = [np.clip(factor, mp_gap_tol_u_l, mp_gap_tol_u)]
        print('tolerance', factor)

        return factor, state
    
class RandomPolicy:
    def __init__(self, env, K=11):
        self.K = K
        self.env = env
    def predict(self, obs_norm, deterministic=True):
        state = 0
        raw_idx = np.random.choice([i for i in range(self.K)])
        raw = -1.0 + 2.0 * raw_idx / float(self.K - 1)
        mp_gap_tol_u = 0.3
        mp_gap_tol_u_l = 1e-3
        print('raw index:', raw)

        scale = np.sqrt(self.env.obs_rms.var + self.env.epsilon)
        obs_true = obs_norm * scale + self.env.obs_rms.mean
        true_gap = obs_true[0][9]
        print('true gap:', true_gap)

        factor = mp_gap_tol_u_l + (raw + 1.0) * 0.5 * (true_gap - mp_gap_tol_u_l)
        # Clip to original bounds
        factor = [np.clip(factor, mp_gap_tol_u_l, mp_gap_tol_u)]
        print('tolerance', factor)

        return factor, state
    
class ExponentialPolicy:
    def __init__(self, env, gap_init, factor): 
        self.gap_init = gap_init
        self.factor = factor
        self.env = env

    def predict(self, obs_norm, deterministic=True):

        scale = np.sqrt(self.env.obs_rms.var + self.env.epsilon)
        obs_true = obs_norm * scale + self.env.obs_rms.mean
        itr = obs_true[0][6]
        gap = self.gap_init * self.factor**itr

        return np.array([[gap]]), 0