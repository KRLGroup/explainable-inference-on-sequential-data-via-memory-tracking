import torch
import torch.nn as nn
from core.dnc.SimplifiedDNC import SimplifiedDNC


class ClozeModel(nn.Module):

    def __init__(self, controller_config, memory_config, embedding_dim, embed_len ,
                    dropout=0.2):
            super(ClozeModel, self).__init__()
            assert controller_config.input_size == embedding_dim

            # flags
            self._num_flags = 3
            self._word_size = embedding_dim+self._num_flags
            self.flag_story = torch.nn.parameter.Parameter(torch.as_tensor([[[1, 0, 0]]], dtype=torch.float32),requires_grad=False)
            self.flag_o1 = torch.nn.parameter.Parameter(torch.tensor([0, 1, 0], dtype=torch.float32),requires_grad=False)
            self.flag_o2 = torch.nn.parameter.Parameter(torch.tensor([0, 0, 1], dtype=torch.float32),requires_grad=False)
            self.query = torch.nn.parameter.Parameter(torch.zeros((1, 1, self._word_size)),requires_grad=False)


            # layers
            self.embeddings = nn.Embedding(embed_len, embedding_dim, padding_idx=0)
            self.dnc = SimplifiedDNC(controller_config, memory_config, output_dim=memory_config.word_size,
                                    clip_value=controller_config.clip_value, dropout=dropout)
            self.fully_connected_layer = nn.Linear(memory_config.word_size*2, 2)

    def forward(self, story_batch, answers):
        option1_batch = answers[0]
        option2_batch = answers[1]
    
        batch_size = len(story_batch)
    
        # embedding
        story_embeds = self.embeddings(story_batch)
        option1_embeds = self.embeddings(option1_batch)
        option2_embeds = self.embeddings(option2_batch)
    
        # add flags
        story_vector = torch.cat([story_embeds,self.flag_story.repeat(batch_size,story_batch.shape[1],1)],2)
        o1_vector =  torch.cat([option1_embeds,self.flag_o1.repeat(batch_size,option1_batch.shape[1],1)],2)
        o2_vector = torch.cat([option2_embeds,self.flag_o2.repeat(batch_size,option2_batch.shape[1],1)],2)
    
        # #add query
        query = self.query.repeat(batch_size,1,1)
        o1_vector = torch.cat((o1_vector, query), 1)
        o2_vector = torch.cat((o2_vector, query), 1)

        # feed the data
        initial_state = self.dnc.initial_state(batch_size)
        story_vector, story_state, hr_story, hw_story = self.dnc(story_vector, initial_state)
        answer1_out, _, hr_answer1, hw_answer1 = self.dnc(o1_vector, story_state)
        answer2_out, _, hr_answer2, hw_answer2 = self.dnc(o2_vector, story_state)
        combined = torch.cat((answer1_out, answer2_out), 1)
        out = self.fully_connected_layer(combined)
        return out, [hr_story, hr_answer1, hr_answer2], [hw_story, hw_answer1, hw_answer2]
