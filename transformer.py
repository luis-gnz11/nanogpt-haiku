import haiku as hk
import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, List, Optional

class HParams(NamedTuple):
  max_seq_len:int
  learning_rate:float
  d_model:int
  embeddings_dropout:float
  h:int  # number of heads in multi-head attention
  encoder_N:int  # number of encoder sublayers
  decoder_N:int  # number of decoder sublayers

# NOTE: h must be a factor of d_model
HPARAMS = HParams(
  max_seq_len = 128,  # T (time)
  learning_rate = 3e-4,
  d_model = 384,  # C (channels)
  embeddings_dropout=0.2,
  h = 6,
  encoder_N = 1,
  decoder_N = 6
  )

# references:
# https://www.youtube.com/watch?v=1biZfFLPRSY
# https://kikaben.com/transformers-positional-encoding/
class VanillaPosEncoding(hk.Module):
  def __init__(self):
    super().__init__()
    hk.get_state("pos_encodings", shape=(HPARAMS.max_seq_len, HPARAMS.d_model), dtype=jnp.float16, init=hk.initializers.Constant(0))
    rows = jnp.repeat(jnp.arange(0, HPARAMS.d_model, 2),2)[0:HPARAMS.d_model]
    cols = jnp.arange(HPARAMS.max_seq_len)
    rows1, cols1 = jnp.meshgrid(rows, cols)
    pos_encodings1 = cols1 / jnp.power(1000, rows1/HPARAMS.d_model)
    indices = self._create_indices(pos_encodings1.shape)
    f_switch = lambda i, x: (1-i) * jnp.sin(x) + i * jnp.cos(x)
    hk.set_state("pos_encodings", f_switch(indices, pos_encodings1))

  def _create_indices(self, size:Tuple[int,int]) -> jax.Array:
    # size is a tuple of (rows, columns) for the output matrix
    arr = a = jnp.array([0, 1])  # arr is a 1D jax array to repeat
    rows, columns = size
    repeats = columns // len(arr)  # how many times to repeat arr along axis 1
    mat = jnp.tile(arr, (rows, repeats))  # create a matrix by tiling arr
    if columns % len(arr) != 0:  # if there are extra columns left
      extra = columns % len(arr)  # calculate how many extra columns are needed
      extra_col = jnp.expand_dims(arr[:extra], axis=0)
      extra_col1 = jnp.repeat(extra_col, rows, axis=0)
      mat = jnp.hstack((mat, extra_col1))  # append the first extra elements of arr to mat
    return mat
    
  def __call__(self, x:jax.Array) -> jax.Array: return hk.get_state("pos_encodings")

class Embed(hk.Module):
  def __init__(self, vocab_size:int, name:str):
    super().__init__()
    self._embed = hk.Embed(
      vocab_size=vocab_size,
      embed_dim=HPARAMS.d_model,
      w_init=hk.initializers.RandomNormal(stddev=1.0),
      lookup_style=hk.EmbedLookupStyle(1),
      name=name,
      precision=jax.lax.Precision('bfloat16'))
          
  def __call__(self, x: jax.Array) -> jax.Array:
    x = self._embed(x)
    return x * jax.lax.sqrt(float(HPARAMS.d_model))

def he_init() -> hk.initializers.Initializer:
  return hk.initializers.VarianceScaling(scale=2.0, mode="fan_in", distribution="truncated_normal")

def encoder_layer(x: jax.Array) -> jax.Array:
  x = hk.MultiHeadAttention( num_heads=HPARAMS.h
                               , key_size=int(HPARAMS.d_model / HPARAMS.h)
                               , w_init=he_init())(x,x,x)
  x = x + hk.LayerNorm(axis=-1, create_scale=False, create_offset=False)(x)
  x = hk.nets.MLP([HPARAMS.d_model, HPARAMS.d_model])(x)
  x = x + hk.LayerNorm(axis=-1, create_scale=False, create_offset=False)(x)
  return x

def encoder(x:jax.Array) -> List[jax.Array]:
  hs = []
  for _ in range(HPARAMS.encoder_N):
    x = encoder_layer(x)
    hs.append(x)
  return hs

def cross_attention_block(x: jax.Array, context:jax.Array) -> jax.Array:
  x = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False)(x)
  x = x + hk.MultiHeadAttention( num_heads=HPARAMS.h
                               , key_size=int(HPARAMS.d_model / HPARAMS.h)
                               , w_init=he_init())(x, context, context)
  return x

def decoder_layer(x: jax.Array, context:Optional[jax.Array]) -> jax.Array:
  B, H, T = x.shape[0], HPARAMS.h, x.shape[1]
  mask=jnp.tril(jnp.ones((T, T)))
  mask_expanded = jnp.repeat(mask[jnp.newaxis, jnp.newaxis], B * H, axis=0)
  mask_expanded = mask_expanded.reshape(B, H, T, T)
  x = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False)(x)
  x = x + hk.MultiHeadAttention( num_heads=HPARAMS.h
                               , key_size=int(HPARAMS.d_model / HPARAMS.h)
                               , w_init=he_init())(x,x,x,mask=mask_expanded)
  x = cross_attention_block(x, context) if context is not None else x
  x = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False)(x)
  x = x + hk.nets.MLP([HPARAMS.d_model, HPARAMS.d_model], activate_final=False)(x)  # two linear transformations with a ReLU activation in between
  return x

def decoder(x: jax.Array, context:List[jax.Array]) -> jax.Array:
  for i in range(HPARAMS.decoder_N):
    x = decoder_layer(x, context[i]) if i < len(context) else decoder_layer(x, None)
  return x
  
def final_block(vocab_size:int, x: jax.Array) -> jax.Array:
  x = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False)(x)
  x = hk.Linear(vocab_size, w_init=he_init())(x)
  return x

# [B, T] -> [B, T, vocab_size]
def f(use_encoder:bool, is_training:bool, vocab_size:int, x: jax.Array) -> jax.Array:
  x = Embed(vocab_size, "embedding")(jnp.asarray(x, dtype=jnp.int32)) \
    + Embed(vocab_size, "positional_encoding_embedding")(jnp.asarray(x, dtype=jnp.int32))
  x = hk.dropout(hk.next_rng_key(), HPARAMS.embeddings_dropout, x) if is_training else x
  x = x * (1-HPARAMS.embeddings_dropout) if is_training else x  # because the haiku dropout doesn't scale the output
  context = encoder(x) if use_encoder else []
  x = decoder(x, context)
  logits = final_block(vocab_size, x)
  return logits
