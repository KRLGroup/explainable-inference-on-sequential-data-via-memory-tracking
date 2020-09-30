from pydantic import BaseModel

class ControllerConfig(BaseModel):
    input_size: int
    lstm_size: int
    num_layers: int
    clip_value: int

class MemoryConfig(BaseModel):
    memory_size: int
    word_size: int
    write_heads: int
    read_heads: int

class TrainingConfig(BaseModel):
  batch_size: int
  dropout: float
  max_grad_norm: int
  learning_rate: float