import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np


class LanguageNetEncoder(nn.Module):
    def __init__(self , args):
        super(LanguageNetEncoder, self).__init__()
        if args.intention == True:
            self.embedding_layer = nn.Embedding(5, 4096).cuda()
            layer_sizes = [4096 , 2048 , 1024]
        else:
            layer_sizes = [4096 , 2048 , 1024]
        self.MLP = nn.Sequential().cuda()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

    def dim_change(self, language_embeddings):

        language_embeddings_list = []

        for embedding in language_embeddings:
            x = embedding.sum(0)
            # x = x.squeeze(0)
            language_embeddings_list.append(x)
        return torch.stack(language_embeddings_list)

    def forward(self, obj_language):
        # model,tokenizer = generator  # llava
        # inputs = tokenizer(
        #     obj_language, padding=True, truncation=True, return_tensors="pt").input_ids.to("cuda")
        # language_embeddings = model.encoder(
        #     inputs).last_hidden_state  # torch.Size([1, 38, 4096])
        # # language_embeddings = generator.generate(obj_language , 200)   llama
        # obj_language_features = self.dim_change(language_embeddings).to("cuda")
        obj_language = obj_language.cuda().long()
        obj_language = self.embedding_layer(obj_language)
        out = self.MLP(obj_language.to(torch.float32)).cuda()
        return out


if __name__ == '__main__':
    pass
