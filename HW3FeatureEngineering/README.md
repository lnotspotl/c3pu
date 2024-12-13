Code to import additional feature sets (Big4, HC&SALM, and 3Tags) into PARROT to be trained and evaluated on. 
Also contains model_config.json confiurations to perform different embeddings (dynamic, byte, and small byte)
of the features. 

* Run and evaluate with train.sh and eval.sh respectively.

* When changing embedding types, model.py must be changed to refelct the embedding type and change in input 
dimensions to the LSTM cell. The following lines must be changed to implement these changes:

in __init__():
self._lstm_cell = nn.LSTMCell(pc_embedder.embed_dim + address_embedder.embed_dim + 4*`EMBEDDING_DIMENSION`, lstm_hidden_size)
#                                                     change embdding dimension here ^^^^^^^^^^^^^^^^^^

in forward():
access_type = torch.tensor([int(cache_access.features[4]) for cache_access in cache_accesses]).unsqueeze(-1)
# for integer embedding ^^^^^^^^^
access_type_embedding = self._pc_embedder([int(cache_access.features[4]) for cache_access in cache_accesses])
# change embedding type here ^^^^^^^^^^^^^




