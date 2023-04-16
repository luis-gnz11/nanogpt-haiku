import haiku as hk
import jax
from jax import random, Array
from jax.nn import log_softmax, one_hot
import optax
import jax.numpy as jnp
from transformer import f, HPARAMS
from typing import Tuple, NamedTuple, List, Optional, Union
from functools import partial
from aim import Run
import pickle

class Config(NamedTuple):
  B:int  # batch size
  max_iters:int
  prng_seed:int
  source_file:str
  last_i:int  # last iteration, used to restore a previous training run, use 0 to start from scratch
  last_picklefile_run_uid:Optional[str]  # uid of the last run that generated a pickle file, used to restore a previous training run, use None to start from scratch

CONFIG = Config(
  B = 32,
  max_iters = 15000,
  prng_seed = 6436,
  source_file = 'novelas_ejemplares.txt',
  last_i = 0,
  last_picklefile_run_uid = None
)

class TrainingState(NamedTuple):
  params: Union[hk.Params, optax.Params]
  state: hk.State
  optimizer_state: optax.OptState
  prng_key: jax.random.PRNGKeyArray
  current_train_loss: Array
  current_val_loss: Array
  min_val_loss: Array
  min_val_step: int

# poor man's data loader
def get_batch(batch_size:int, max_seq_len:int, key, data:Array) -> Tuple[Array, Array]:
  ix = random.randint(key, shape=(batch_size,), minval=0, maxval=len(data) - max_seq_len)
  x = jnp.stack([data[i:i+max_seq_len] for i in ix])
  y = jnp.stack([data[i+1:i+1+max_seq_len] for i in ix])
  return (x, y)

optimizer = optax.adamw(HPARAMS.learning_rate)
f1 = hk.transform_with_state(partial(f, False, False, 88))
f2 = jax.jit(f1.apply)

# cross entropy loss
@jax.jit
def loss_f(params:hk.Params, state:hk.State, key:jax.random.PRNGKeyArray, x:Array, y:Array) -> Array:
  logits, _ = f2(params, state, key, x)
  yhat_log_probs = log_softmax(logits, axis=-1)
  y_one_hot = one_hot(y, num_classes=logits.shape[-1])
  loss = -jnp.sum(yhat_log_probs * y_one_hot, axis=-1)
  mean_loss = jnp.mean(loss)
  return mean_loss

def train_step(tst:TrainingState, x:Array, y:Array) -> TrainingState:
  key1, key2 = jax.random.split(tst.prng_key, 2)
  loss_value, grads = jax.value_and_grad(loss_f)(tst.params, tst.state, key2, x, y)
  updates, new_opt_state = optimizer.update(grads, tst.optimizer_state, tst.params)
  new_params = optax.apply_updates(tst.params, updates)
  return TrainingState(
    new_params,
    tst.state,
    new_opt_state,
    key1,
    loss_value,
    tst.current_val_loss,
    tst.min_val_loss,
    tst.min_val_step
    )

def eval_step(tst:TrainingState, i:int, x:Array, y:Array) -> TrainingState:
  key1, key2 = jax.random.split(tst.prng_key, 2)
  val_loss = loss_f(tst.params, tst.state, key2, x, y)
  return TrainingState(
    tst.params,
    tst.state,
    tst.optimizer_state,
    key1,
    tst.current_train_loss,
    val_loss,
    val_loss if val_loss < tst.min_val_loss else tst.min_val_loss,
    i if val_loss < tst.min_val_loss else tst.min_val_step
    )

def aim_log_(run:Run, i:int, tst:TrainingState) -> None:
  run.track(float(tst.current_train_loss), name='loss', step=i, context={ "subset":"train" })
  run.track(float(tst.current_val_loss), name='loss', step=i, context={ "subset":"val" })
  pass

def stdout_log_(i:int, mod:int, tst:TrainingState) -> None:
  if i % mod == 0:
    print(f"step {i}: train loss {float(tst.current_train_loss):.4f}, val loss {float(tst.current_val_loss):.4f}, min val loss {float(tst.min_val_loss):.4f} (step {tst.min_val_step})")

def save_params_(i:int, pickle_filename:str, tst:TrainingState) -> None:
  if i % 100 == 0:
    with open(pickle_filename, "wb") as f: pickle.dump(tst.params, f)
  if tst.current_val_loss == tst.min_val_loss:
    with open(f"{pickle_filename}.min_val.pkl", "wb") as f: pickle.dump(tst.params, f)

def logs_(run, i:int, tst:TrainingState, mod:int) -> None:
  aim_log_(run, i, tst)
  stdout_log_(i, mod, tst)

def train_(pickle_filename:str, tst:TrainingState, train_data:Array, eval_data:Array, run:Run) -> TrainingState:
  batch_keys = jax.random.split(jax.random.PRNGKey(CONFIG.prng_seed), CONFIG.max_iters)
  for i in range(CONFIG.last_i, CONFIG.max_iters):
    xs, ys = get_batch(CONFIG.B, HPARAMS.max_seq_len, batch_keys[i], train_data)
    tst = train_step(tst, xs, ys)
    xs1, ys1 = get_batch(100, HPARAMS.max_seq_len, batch_keys[i], eval_data)
    tst = eval_step(tst, i, xs1, ys1)
    save_params_(i, pickle_filename, tst)
    logs_(run, i, tst, 10)
  return tst

def print_run_info(aimlogger, vocab_size:int, chars:List[str]) -> None:
  print("jax devices:", jax.devices())
  print("run uid:", aimlogger.hash)  
  print("vocab_size:", vocab_size, "chars:", chars)
  print("config:", dict(CONFIG._asdict()))
  print("---")

def main_() -> None:
  aimlogger = Run(run_hash=CONFIG.last_picklefile_run_uid) if CONFIG.last_picklefile_run_uid is not None else Run()
  aimlogger["config"] = dict(CONFIG._asdict())
  aimlogger["hparams"] = dict(HPARAMS._asdict())

  with open(CONFIG.source_file, "r", encoding="utf-8") as fdata: dataset_txt = fdata.read()

  # here are all the unique characters that occur in this text
  chars = sorted(list(set(dataset_txt)))
  vocab_size = len(chars)
  
  print_run_info(aimlogger, vocab_size, chars)

  # create a mapping from characters to integers
  stoi = { ch:i for i,ch in enumerate(chars) }
  itos = { i:ch for i,ch in enumerate(chars) }
  encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
  decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

  # train and test splits
  data = jnp.asarray(encode(dataset_txt), dtype='bfloat16')
  n = int(0.9*len(data)) # first 90% will be train, rest val
  train_data = data[:n]
  val_data = data[n:]

  key1, key2 = jax.random.split(jax.random.PRNGKey(CONFIG.prng_seed), 2)
  init_params, init_state = f1.init(key1, jnp.zeros((CONFIG.B, HPARAMS.max_seq_len), dtype='bfloat16'))
  init_optimizer_state = optimizer.init(init_params)
  if CONFIG.last_picklefile_run_uid is not None:
    with open(f"params/params.{CONFIG.last_picklefile_run_uid}.pkl", "rb") as f: init_params = pickle.load(f)

  tst:TrainingState = TrainingState (
    init_params,
    init_state,
    init_optimizer_state,
    key2,
    jnp.array([], dtype='bfloat16'),
    jnp.array([], dtype='bfloat16'),
    jnp.array([float("inf")], dtype='bfloat16'),
    0
  )
  tst:TrainingState = train_(f"params/params.{aimlogger.hash}.pkl", tst, train_data, val_data, aimlogger)

main_()
