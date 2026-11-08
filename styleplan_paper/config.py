"""The attached paper's configuration; all architectural choices are explicit."""
from dataclasses import asdict, dataclass
import math

ABLATIONS = ('full', 'no_partner_model', 'style_only', 'intent_only', 'concat',
             'no_arbitration', 'no_style', 'no_intent')

@dataclass(frozen=True)
class ModelConfig:
    obs_dim: int
    task_dim: int
    global_state_dim: int = 0
    num_actions: int = 6
    style_dim: int = 32
    intent_dim: int = 16
    history_window: int = 50
    intent_window: int = 10
    prediction_horizon: int = 5
    hidden_dim: int = 64
    context_dim: int = 64
    ema_decay: float = 0.99
    ablation: str = 'full'
    style_injection: str = 'bounded'
    slow_injection_scale: float = 0.03

    def __post_init__(self):
        for key in ('obs_dim', 'task_dim', 'num_actions', 'style_dim', 'intent_dim',
                    'history_window', 'intent_window', 'hidden_dim', 'context_dim'):
            if getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive')
        if self.global_state_dim < 0 or self.prediction_horizon < 0:
            raise ValueError('global_state_dim and horizon must be nonnegative')
        if self.intent_window > self.history_window:
            raise ValueError('intent_window cannot exceed history_window')
        if not 0 <= self.ema_decay < 1 or self.ablation not in ABLATIONS:
            raise ValueError('Invalid EMA decay or ablation')
        if self.style_injection not in ('bounded', 'legacy'):
            raise ValueError('style_injection must be bounded or legacy')
        if not math.isfinite(self.slow_injection_scale) or self.slow_injection_scale < 0:
            raise ValueError('slow_injection_scale must be finite and nonnegative')

    @property
    def prediction_steps(self):
        # Literal Eq.9: sum_{k=0}^{H}; H=5 therefore has six distributions.
        return self.prediction_horizon + 1

    @property
    def fingerprint_dim(self):
        return self.num_actions + 13

    def to_dict(self):
        return asdict(self)

@dataclass(frozen=True)
class TrainConfig:
    num_envs: int = 8
    num_steps: int = 64
    epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    intent_coef: float = 0.1
    pretrain_intent_coef: float = 1.0
    style_coef: float = 0.0
    prior_coef: float = 0.01
    smoothness_coef: float = 0.01
    max_grad_norm: float = 0.5
    contrastive_temperature: float = 0.1

    def __post_init__(self):
        if min(self.num_envs, self.num_steps, self.epochs) <= 0:
            raise ValueError('Batch dimensions and epochs must be positive')
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError('learning_rate and max_grad_norm must be positive')
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError('Invalid discount or GAE lambda')
        if self.clip_epsilon <= 0 or self.contrastive_temperature <= 0:
            raise ValueError('Clip epsilon and contrastive temperature must be positive')
        coefficients = (self.value_coef, self.entropy_coef, self.intent_coef,
                        self.pretrain_intent_coef, self.style_coef,
                        self.prior_coef, self.smoothness_coef)
        if any(not math.isfinite(x) or x < 0 for x in coefficients):
            raise ValueError('Loss coefficients must be finite and nonnegative')
