import torch
import torch.nn as nn


_EPSILON = 1e-6

def exclusive_cumprod(x):
    """Exclusive cumulative product [a, b, c] => [1, a, a * b]
    *from https://github.com/j-min/MoChA-pytorch/blob/master/attention.py
    """
    batch_size, sequence_length = x.size()
    if torch.cuda.is_available():
        one_x = torch.cat([torch.ones(batch_size, 1).cuda(), x], dim=1)[:, :-1]
    else:
        one_x = torch.cat([torch.ones(batch_size, 1), x], dim=1)[:, :-1]
    return torch.cumprod(one_x, dim=1)

class ContentAddressing(nn.Module):
    def __init__(self):
        super(ContentAddressing, self).__init__()
        self.softmax = nn.Softmax(2)

    def _vector_norms(self, v):
        """ Computes the vector norms

        Args:
            v: The vector from which there must be calculated the norms

        Returns:
             A tensor containing the norms of input vector v
        """

        squared_norms = torch.sum(v * v, dim=2, keepdim=True)
        return torch.sqrt(squared_norms + _EPSILON)

    def _weighted_softmax(self, similarity, strengths):
        """ Computes the vector norms

        Args:
            similarity:  A tensor of shape `[batch_size, num_heads, memory_size]`
            strengths: A tensor of shape `[batch_size, num_read_heads]`
        Returns:
             A tensor of same shape of similarity
        """
        transformed_strengths = strengths.unsqueeze(-1)
        sharp_activations = similarity * transformed_strengths
        return self.softmax(sharp_activations)

    def forward(self, memory, keys, strengths):
        """ Cosine-weighted attention.
        Calculates the cosine similarity between a query and each word in memory, then
        applies a weighted softmax to return a sharp distribution.

        Args:
            memory: A tensor of shape [batch_size, num_rows, num_cols]
            keys: A tensor of shape [batch_size, num_read_heads, num_cols]
            strengths: A tensor of shape [batch_size,num_read_heads]

        Returns:
            a tensor indicating the cosine similarity between the keys and the
            memory content weighted by the strengths vector
        """

        # transpose instead of conj because conj is no ops for real matrix
        dot = torch.matmul(keys, torch.transpose(memory, 1, 2))
        memory_norms = self._vector_norms(memory)
        key_norms = self._vector_norms(keys)
        norm = torch.matmul(key_norms, torch.transpose(memory_norms, 1, 2))
        sim = dot / (norm + _EPSILON)
        return self._weighted_softmax(sim, strengths)


class Memory(nn.Module):
    """Memory module of the Simplified Differentiable Neural Computer.
     It is a modified version of the original Access Module
     (https://github.com/deepmind/dnc/blob/master/dnc/access.py)
     This memory module supports multiple read and write heads.
     Write-address selection is done using unused memory.
     Read-address selection is done using content-based lookup.
     """
    def __init__(self, memory_config,input_size):
        """Creates a Memory module.
        Args:
            memory_config: Dictionary containing the desired configuration of
                    memory. The dict should contain the following keys:
                    write_heads, read_heads, input_dim, memory_size and word_size
        """
        super(Memory, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_write_heads = memory_config.write_heads
        self.num_read_heads = memory_config.read_heads
        self.input_size = input_size
        self.num_rows = memory_config.memory_size
        self.num_cols = memory_config.word_size
        self.content_attention = ContentAddressing()

        # Declaration of layers to compute interface vector
        self.write_vectors = nn.Linear(self.input_size, self.num_write_heads * self.num_cols)
        self.erase_vectors = nn.Linear(self.input_size, self.num_write_heads * self.num_cols)
        self.write_gate = nn.Linear(self.input_size, self.num_write_heads)
        self.read_keys = nn.Linear(self.input_size, self.num_cols * self.num_read_heads)
        self.read_strengths = nn.Linear(self.input_size, self.num_read_heads)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.free_gate = nn.Linear(self.input_size,self.num_read_heads)

        # Weights initialization
        nn.init.xavier_normal_(self.write_vectors.weight)
        nn.init.xavier_normal_(self.erase_vectors.weight)
        nn.init.xavier_normal_(self.write_gate.weight)
        nn.init.xavier_normal_(self.read_keys.weight)
        nn.init.xavier_normal_(self.read_strengths.weight)
        nn.init.zeros_(self.write_vectors.bias)
        nn.init.zeros_(self.erase_vectors.bias)
        nn.init.zeros_(self.write_gate.bias)
        nn.init.zeros_(self.read_keys.bias)
        nn.init.zeros_(self.read_strengths.bias)

    def batch_gather(self, values, indices):
        """Returns batched `gather` for every row in the input. Equivalent of batch_gather function of
         original implementation"""

        idx = indices.unsqueeze(-1)
        idx = idx.int()
        size = indices.shape[0]
        rg = torch.arange(size, dtype=torch.int32).to(self.device)
        rg = rg.unsqueeze(-1)
        rg = rg.repeat([1, int(indices.shape[-1])])
        rg = rg.unsqueeze(-1)
        gidx = torch.cat((rg, idx), -1)
        gidx = gidx.long()
        out = values[gidx[:, :, 0], gidx[:, :, 1]]
        return out

    def _erase_and_write(self, memory, write_weights, write_vectors, erase_vectors):
        """Module to erase and write in the external memory.

        Erase operation:
          M_t'(i) = M_{t-1}(i) * (1 - w_t(i) * e_t)

        Add operation:
          M_t(i) = M_t'(i) + w_t(i) * a_t

        where e are the reset_weights, w the write weights and a the values.

        Args:
            memory: 3-D tensor of shape `[batch_size, num_rows, num_cols]`.
            write_weights: 3-D tensor `[batch_size, num_write_heads, memory_size]`.
            write_vectors: 3-D tensor `[batch_size, num_write_heads, num_cols]`.
            erase_vectors: 3-D tensor `[batch_size, num_write_heads, num_cols]`.

        Returns:
            3-D tensor of shape `[batch_size, num_write_heads, num_cols]`.
        """

        extended_write_weights = write_weights.unsqueeze(3)
        extended_erase_vectors = erase_vectors.unsqueeze(2)
        weighted_resets = extended_erase_vectors * extended_write_weights
        reset_gate = torch.prod(1 - weighted_resets, 1)
        write_memory = torch.matmul(torch.transpose(write_weights, 1, 2), write_vectors)
        updated_memory = (memory * reset_gate) + write_memory

        return updated_memory

    def _update_usage(self, usage, write_weights):
        """Calculates the new memory usage u_t.

        Memory that was written to in the previous time step will have its usage
        increased; Respect to the original paper here we don't decrease the memory
        usage of the cells that were read from .

        Args:
            usage: tensor of shape `[batch_size, num_rows]` giving
                    usage u_{t - 1} at the previous time step, with entries in range
                    [0, 1].
            write_weights: tensor of shape `[batch_size, num_write_heads,num_rows]`
                    giving write weights at previous time step.

        Returns:
            tensor of shape `[batch_size, num_rows]` representing updated memory
            usage.
        """
        with torch.no_grad():
            write_weights = 1 - torch.prod(1 - write_weights, 1)
            updated_usage = usage + (1 - usage) * write_weights
            return updated_usage

    def _allocate(self, usage_vector):
        """Computes allocation by sorting `usage`.

        Args:
            usage_vector: tensor of shape `[batch_size, memory_size]` indicating current
                    memory usage.

        Returns:
            Tensor of shape `[batch_size, memory_size]` corresponding to allocation.
        """
        # Ensure values are not too small prior to cumprod.
        usage = _EPSILON + (1 - _EPSILON) * usage_vector
        nonusage = 1 - usage
        sorted_nonusage, indices = torch.topk(nonusage, k=self.num_rows)
        sorted_usage = 1 - sorted_nonusage
        prod_sorted_usage = exclusive_cumprod(sorted_usage)
        sorted_allocation = sorted_nonusage * prod_sorted_usage
        _, inverse_indices = torch.topk(indices, k=self.num_rows, largest=False)
        return self.batch_gather(sorted_allocation, inverse_indices)

    def _allocation_weights(self, write_gates, num_writes, usage):
        """Calculates freeness-based locations for writing to.

        This finds unused memory by ranking the memory locations by usage, for each
        write head. (For more than one write head, we use a "simulated new usage"
        which takes into account the fact that the previous write head will increase
        the usage in that area of the memory.)

        Args:
            write_gates: A tensor of shape `[batch_size,num_write_heads]` with values in
                    the range [0, 1] indicating how much each write head does writing
                    based on the address returned here (and hence how much usage
                    increases).
            num_writes: The number of write heads to calculate write weights for.
            usage: A tensor of shape `[batch_size,num_rows]` representing
                    current memory usage.

        Returns:
            tensor of shape `[batch_size, num_writes, memory_size]` containing the
            freeness-based write locations.
        """
        write_gates = write_gates.unsqueeze(-1)
        allocation_weights = []
        for i in range(num_writes):
            allocation = self._allocate(usage)
            allocation_weights.append(allocation)
            usage = usage + ((1-usage)*write_gates[:, i, :]*allocation_weights[i])
        allocation_weights = torch.stack(allocation_weights, 1)
        return allocation_weights

    def _write_weights(self, write_gates, usage):
        """Calculates the memory locations to write to.

        This uses a combination of content-based lookup and finding an unused
        location in memory, for each write head.

        Args:
            write_gates: A tensor of shape `[batch_size,num_write_heads]` with values in
                    the range [0, 1] indicating how much each write head does writing
                    based on the address returned here (and hence how much usage
                    increases).
            usage: A tensor of shape `[batch_size,num_rows]` representing
                    current memory usage.

        Returns:
            tensor of shape `[batch_size, num_writes, memory_size]` indicating where
            to write to (if anywhere) for each write head.
        """

        write_allocation_weights = self._allocation_weights(write_gates, self.num_write_heads, usage)
        write_gate = write_gates.unsqueeze(-1)
        write_weights = write_gate*write_allocation_weights
        return write_weights

    def forward(self, inputs, prev_state):
        """Connects the Memory module into the graph.

        Args:
            inputs: tensor of shape `[batch_size, input_size]`. This is used to
                    control this access module.
            prev_state: Instance of `AccessState` containing the previous state.

        Returns:
            A tuple `(output, next_state)`, where `output` is a tensor of shape
            `[batch_size, num_read_heads, num_cols]`, and `next_state` is the new
            dictionary containing the memory state of the current time t.
        """

        # get interface vectors
        erase_vector = self.sigmoid(self.erase_vectors(inputs))
        write_gate = self.sigmoid(self.write_gate(inputs))
        write_vector = self.write_vectors(inputs)
        read_keys = self.read_keys(inputs)
        read_strengths = self.softplus(self.read_strengths(inputs))
        write_vector = write_vector.view(-1, self.num_write_heads, self.num_cols)
        erase_vector = erase_vector.view(-1, self.num_write_heads, self.num_cols)
        read_keys = read_keys.view(-1, self.num_read_heads, self.num_cols)

        # write memory
        usage = self._update_usage(prev_state['usage'], prev_state['write_weights'])
        write_weights = self._write_weights(write_gate, usage)
        updated_memory = self._erase_and_write(prev_state['memory'], write_weights, write_vector, erase_vector)

        # read memory based only on content lookup
        read_weights = self.content_attention(prev_state['memory'], read_keys, read_strengths)
        read_vectors = torch.matmul(read_weights, updated_memory)
        memory_state = {
            'memory': updated_memory,
            'write_weights': write_weights,
            'read_weights': read_weights,
            'usage': usage
        }
        return read_vectors, memory_state

    def initial_state(self, batch_size):
        with torch.no_grad():
            memory = torch.zeros(batch_size, self.num_rows, self.num_cols).to(self.device)
            usage = torch.zeros(batch_size, self.num_rows).to(self.device)
            write_weights = torch.zeros([batch_size, self.num_write_heads, self.num_rows],requires_grad=False).to(self.device)
            read_weights = torch.zeros([batch_size, self.num_read_heads, self.num_rows],requires_grad=False).to(self.device)
            return {'memory': memory, 'write_weights': write_weights, 'read_weights': read_weights, 'usage': usage}
