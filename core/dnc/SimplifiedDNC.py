import torch.nn as nn
import torch
from .memory import Memory


class SimplifiedDNC(nn.Module):
    """Simplified Differentiable Neural Computer.
    Contains controller and memory access module.
    """
    def __init__(self, controller_config, memory_config, output_dim, clip_value=None, dropout=0.2):
        """Initializes the Simplified DNC.
           Args:
             controller_config: dictionary of controller (LSTM) module configurations.
             memory_config: dictionary of memory module configurations.
             output_dim: output dimension size of DNC.
             clip_value: clips controller and DNC output values to between
                 `[-clip_value, clip_value]` if specified.
             dropout: dropout to apply for bypass dropout.
           """
        super(SimplifiedDNC, self).__init__()

        self.output_dim = output_dim
        self.clip_value = clip_value or 0
        self.dropout = nn.Dropout(p=dropout)

        # layers
        self.controller = nn.LSTMCell(input_size=controller_config.input_size,
                                      hidden_size=controller_config.lstm_size)
        self.linear = nn.Linear(controller_config.lstm_size+
                            memory_config.word_size*memory_config.read_heads,
                                self.output_dim)
        self.memory = Memory(memory_config,controller_config.lstm_size)
        self.layer_norm = nn.LayerNorm(controller_config.lstm_size)
        
        # initializations
        nn.init.xavier_uniform_(self.controller.weight_hh)
        nn.init.xavier_uniform_(self.controller.weight_ih)


    def forward(self, inputs, prev_state):
        """Forward call of Simplified DNC.
        Args:
          inputs: Tensor input.
          prev_state: A tuple containing the fields `memory_state`,
               and `memory_state`.
              `memory_state` is a tuple of the access module's state, and
              `controller_state` is a tuple of controller module's state.
        Returns:
          A tuple `(output, state, read_history, write_history)` where `output`
          is the output tensor of the last step, `state`
          is the last `DNC State` tuple containing the fields `memory_state`,
          and `controller_state`, read_history is a tensor containing the
          read weights for each sequence's step and the write_history is a
          tensor containing the write weights for each step of the sequence.
        """
        output = []
        state = {
            'controller_state': prev_state['controller_state'],
            'memory_state': prev_state['memory_state']
        }
        steps = inputs.shape[1]
        batch_size = inputs.shape[0]
        batch_history_read = torch.zeros((batch_size, steps, self.memory.num_read_heads, self.memory.num_rows))
        batch_history_write = torch.zeros((batch_size, steps, self.memory.num_write_heads, self.memory.num_rows))

        for i in range(steps):
            controller_state = self.controller(inputs[:, i, :], state['controller_state'])

            controller_output = controller_state[0]

            read_vector, memory_state = self.memory(self.layer_norm(self._clip_if_enabled(controller_output)), state['memory_state'])
            state = {
                'controller_state': controller_state,
                'memory_state': memory_state
            }

            for batch in range(batch_size):
                batch_history_read[batch][i] = memory_state['read_weights'][batch]
                batch_history_write[batch][i] = memory_state['write_weights'][batch]

            dropped_controller_output = self.dropout(controller_output)
            read_vector = torch.flatten(read_vector, start_dim=1)
            input_final_layer = torch.cat((dropped_controller_output, read_vector), 1)
            final_output = self.linear(input_final_layer)
            output.append(final_output)
        
        # we are interested only on the last output of the sequence
        out = output[-1]
        return out, state, batch_history_read, batch_history_write

    def _clip_if_enabled(self, x):
        if self.clip_value > 0:
            return torch.clamp(x, -self.clip_value, self.clip_value)
        else:
            return x

    def initial_state(self, batch_size):
        return {
            'controller_state': None,
            'memory_state': self.memory.initial_state(batch_size)
        }


