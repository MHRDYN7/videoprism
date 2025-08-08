import jax
import jax.numpy as jnp
import mediapy
import numpy as np
from PIL import Image
import videoprism.models as vp


def read_and_preprocess_video(
    filename: str, target_num_frames: int, target_frame_size: tuple[int, int]
):
  """Reads and preprocesses a video."""

  frames = mediapy.read_video(filename)

  # Sample to target number of frames.
  frame_indices = np.linspace(
      0, len(frames), num=target_num_frames, endpoint=False, dtype=np.int32
  )
  frames = np.array([frames[i] for i in frame_indices])

  # Resize to target size.
  original_height, original_width = frames.shape[-3:-1]
  target_height, target_width = target_frame_size
  assert (
      original_height * target_width == original_width * target_height
  ), 'Currently does not support aspect ratio mismatch.'
  frames = mediapy.resize_video(frames, shape=target_frame_size)

  # Normalize pixel values to [0.0, 1.0].
  frames = mediapy.to_float01(frames)

  return frames

  
MODEL_NAME = 'videoprism_public_v1_base_hf'
NUM_FRAMES = 16
FRAME_SIZE = 288

flax_model = vp.get_model(MODEL_NAME)  #? step 1 get model is in models
loaded_state = vp.load_pretrained_weights(MODEL_NAME)    #? step 4  flax.core.frozen_dict.FrozenDict
print(type(loaded_state))

#?@jax.jit

def forward_fn(inputs, train=False):
  return flax_model.apply(loaded_state, inputs, train=train)   #? inputs is a jnp array of shape (B, 16, 288, 288, 3)

VIDEO_FILE_PATH = (
    './videoprism/assets/water_bottle_drumming.mp4'
)
frames = read_and_preprocess_video(
    VIDEO_FILE_PATH,
    target_num_frames=NUM_FRAMES,
    target_frame_size=[FRAME_SIZE, FRAME_SIZE],
)
#print("frames read")
#mediapy.show_video(frames, fps=6.0)

frames = jnp.asarray(frames[None, ...])  # Add batch dimension.
#print(f'Input shape: {frames.shape}')

embeddings, _ = forward_fn(frames)   #? step 5 use the aply method to invoke the __call__ of FactorizedEncoder
#print(f'Encoded embedding shape: {embeddings.shape}')