"""Safe array-only checkpoints with provenance and refuse-to-overwrite behavior."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from .config import ModelConfig
from .types import RunningNorm, AdamState, TrainState
from .model import init_params


def source_manifest():
    root = Path(__file__).resolve().parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}


def _flatten(value, prefix, dest):
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(child, prefix+'/'+key, dest)
    else:
        dest[prefix] = np.asarray(value)


def _restore(arrays, prefix):
    root = {}
    for key in arrays.files:
        if not key.startswith(prefix+'/'):
            continue
        parts = key[len(prefix)+1:].split('/')
        target = root
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = jnp.asarray(arrays[key])
    return root


def save_checkpoint(path, state, config, metadata=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    arrays = {}
    _flatten(state.params, 'params', arrays)
    _flatten(state.optimizer.m, 'adam_m', arrays)
    _flatten(state.optimizer.v, 'adam_v', arrays)
    arrays.update(norm_count=np.asarray(state.norm.count), norm_mean=np.asarray(state.norm.mean),
                  norm_m2=np.asarray(state.norm.m2), adam_count=np.asarray(state.optimizer.count))
    np.savez_compressed(path/'arrays.npz', **arrays)
    meta = dict(schema_version=3, implementation='attached-paper-StylePlan-JAX-v0.1.2',
        config=asdict(config), source_hashes=source_manifest(), jax_version=jax.__version__,
        array_file_sha256=hashlib.sha256((path/'arrays.npz').read_bytes()).hexdigest(),
        metadata=metadata or {})
    (path/'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return path


def load_checkpoint(path):
    path = Path(path)
    meta = json.loads((path/'metadata.json').read_text())
    if meta.get('schema_version') not in (2, 3):
        raise ValueError('Unsupported checkpoint schema')
    digest = hashlib.sha256((path/'arrays.npz').read_bytes()).hexdigest()
    if digest != meta['array_file_sha256']:
        raise ValueError('Checkpoint content hash does not match metadata')
    config_data = dict(meta['config'])
    if meta['schema_version'] == 2:
        # Missing fields must preserve v0.1.0 dynamics rather than silently
        # enabling the new recurrence when reading a historical checkpoint.
        config_data.setdefault('style_injection', 'legacy')
        config_data.setdefault('slow_injection_scale', 1.0)
    elif not {'style_injection', 'slow_injection_scale'} <= config_data.keys():
        raise ValueError('Schema 3 requires explicit recurrent conditioning settings')
    config = ModelConfig(**config_data)
    template = init_params(jax.random.PRNGKey(0), config)
    with np.load(path/'arrays.npz', allow_pickle=False) as arrays:
        params = _restore(arrays, 'params')
        m, v = _restore(arrays, 'adam_m'), _restore(arrays, 'adam_v')
        for name, tree in [('params', params), ('adam_m', m), ('adam_v', v)]:
            if jax.tree_util.tree_structure(tree) != jax.tree_util.tree_structure(template):
                raise ValueError(name+' structure differs from configured architecture')
            for actual, expected in zip(jax.tree_util.tree_leaves(tree), jax.tree_util.tree_leaves(template)):
                if actual.shape != expected.shape or not np.isfinite(np.asarray(actual)).all():
                    raise ValueError(name+' contains invalid shapes or nonfinite values')
        norm = RunningNorm(*(jnp.asarray(arrays[k]) for k in ('norm_count', 'norm_mean', 'norm_m2')))
        if norm.mean.shape != (config.fingerprint_dim,) or norm.m2.shape != norm.mean.shape:
            raise ValueError('Invalid running-normalization shape')
        if any(not np.isfinite(np.asarray(x)).all() for x in norm) or float(norm.count) < 0:
            raise ValueError('Invalid running-normalization values')
        optimizer = AdamState(jnp.asarray(arrays['adam_count']), m, v)
    state = TrainState(params, norm, optimizer)
    return dict(config=config, params=params, norm=norm, state=state, metadata=meta)
