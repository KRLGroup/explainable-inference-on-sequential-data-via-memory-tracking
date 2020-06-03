import torch

class ExplanationModule():

    def __init__(self, padding_value,top_k):
        self.padding_idx = padding_value
        self.top_k = top_k

    def get_sgt(self,network, background, answers):
        predictions = []
        for element in background:
            outcome, _, _ = network(element, answers)
            predicted = torch.argmax(outcome, 1)
            predictions.append(predicted.item())
        return predictions


    def _parse_write_history(self, write_weights):
        """ Analyzes the history of write weights to extract the main written
        cells for each step.

        Args:
            write_weights: A tensor of shape of shape `[timesteps, batch_size,
                    num_write_heads, memory_size]`

        Returns:
            A list of length = timesteps containing for each at the index `i` the
            main written cells for the element `i` of the sequence.
        """
        cells_timesteps = []
        for t in range(write_weights.shape[0]):
            write_history_t = write_weights[t]
            median_value = torch.median(write_history_t)

            # the epsilon is needed to stabilize the process and avoid that cells with
            # very tiny improvements over median case are considered "special"
            written_cells = (write_history_t > median_value+1e-3).nonzero()[:, 1]
            cells_timesteps.append(written_cells)
        return cells_timesteps

    def _parse_read_history(self, read_weights):
        """ Analyzes the history of read weights to extract the main read
        cells for each step.

        Args:
            read_weights: A tensor of shape of shape `[timesteps, batch_size,
                    num_read_heads, memory_size]`

        Returns:
            A list of length = timesteps containing for each at the index `i` the
            main read cells for the element `i` of the sequence.
        """
        all_cells = []
        for t in range(read_weights.shape[0]):
            read_history_t = read_weights[t]
            starting_step = 0
            for head in range(starting_step, read_history_t.shape[0]):
                history_head = read_history_t[head]
                median = torch.median(history_head)
                top_values, top_indices = torch.topk(history_head, self.top_k)
                above_mean = (top_values > median).nonzero()[:, 0]
                read_cells = top_indices[above_mean]
                all_cells.append(read_cells)
        return all_cells
        
        
    def get_rank(self,network, background, write_history,read_history):
        story_mapping = self._parse_write_history(write_history)
        read_cells = self._parse_read_history(read_history)

        lens = [len(e[0]) for e in background]
        premise_steps = torch.zeros(sum(lens))
        current_index = 0
        current_premise = 1
        for l in lens:
            next_index = current_index + l
            premise_steps[current_index:next_index] = current_premise
            current_premise +=1
            current_index = next_index
        count_answer = {1: 0, 2: 0, 3: 0, 4: 0}
        for time in range(len(read_cells)):
            for cell in read_cells[time]:
                timesteps = [i for i, x in enumerate(story_mapping) if cell in x]
                for t in timesteps:
                    check_time = t - 0
                    premise_check = premise_steps[max(0, check_time)].item()
                    # Don't consider the PAD
                    if premise_check > 0:
                        count_answer.update({premise_check: count_answer.get(premise_check) + 1})

        rank = [k for k, v in sorted(count_answer.items(), reverse=True, key=lambda item: item[1])]
        total_readings = sum(count_answer.values())
        if total_readings > 0:
            percentage = [ (count_answer.get(i)*100)/total_readings for i in rank]
        else:
            percentage = [0,0,0,0]
        return rank, percentage