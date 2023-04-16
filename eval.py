import haiku as hk
import jax
from jax import random
from jax import Array
import jax.numpy as jnp
from transformer import f, HPARAMS
from functools import partial
from typing import Tuple, NamedTuple
import pickle
from time import sleep, time

class Config(NamedTuple):
  B:int  # batch size
  prng_seed:int

CONFIG = Config(
  B = 1,
  prng_seed = int(time())
)

with open("novelas_ejemplares.txt", "r", encoding="utf-8") as fdata: nes = fdata.read()
chars = sorted(list(set(nes)))
vocab_size = len(chars)

# mapping from chars to ints
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

f1 = hk.transform_with_state(partial(f, False, False, vocab_size))
f2 = jax.jit(f1.apply)
key_init, key_loop = jax.random.split(jax.random.PRNGKey(CONFIG.prng_seed), 2)
init_params, init_state = f1.init(key_init, jnp.zeros((CONFIG.B, HPARAMS.max_seq_len), dtype=jnp.int32))

last_picklefile_run_uid = "c1069c8d186b4b4e8852ebfe"
with open(f"params/params.{last_picklefile_run_uid}.pkl.min_val.pkl", "rb") as f: pkl_params = pickle.load(f)
with open(f"eval_context.txt", "rt") as f: init_context = f.read()

print("jax devices:", jax.devices())
print("run hash:", last_picklefile_run_uid)
print("vocab_size:", vocab_size, "chars:", chars)
print(f"context: {init_context}, (size {len(init_context)})")
print("random seed:", CONFIG.prng_seed)
print("param count:", sum(x.size for x in jax.tree_leaves(pkl_params)))
print("Ctrl-C to stop")
print("---\n")

x = jnp.expand_dims(jnp.asarray(encode(init_context), dtype=jnp.int32), axis=0)
print(decode(x[0].tolist()), end='', flush=True)
while True:
  key_loop, key1, key2 = jax.random.split(key_loop, 3)
  logits, _ = f1.apply(pkl_params, init_state, key1, x)
  yh = random.categorical(key2, logits=logits, axis=-1)
  print(decode(yh[0][-1:].tolist()), end='', flush=True)
  x = jax.numpy.expand_dims(jnp.concatenate([x[0], yh[0][-1:]]), axis=0)
  x = x[:, -HPARAMS.max_seq_len:]
  sleep(0.05)

