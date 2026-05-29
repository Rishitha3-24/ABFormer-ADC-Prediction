from math import fabs

AminoAcid_Vocab = {
    "A":1,
    "R":2,
    "N":3, 
    "D":4,
    "C":5,
    "Q":6,
    "E":7,
    "G":8,
    "H":9,
    "I":10,
    "L":11,
    "K":12,
    "M":13,
    "F":14,
    "P":15,
    "S":16,
    "T":17,
    "W":18,
    "Y":19,
    "V":20,

    "X":21,  # <END>

}


class configuration():
    def __init__(self,
                    hidden_size: int = 768,
                    max_position_embeddings: int = 263,
                    type_residue_size: int = 9,
                    layer_norm_eps: float = 1e-12,
                    hidden_dropout_prob = 0.1,
                    use_bias = True,
                    initializer_range=0.02,
                    num_hidden_layers = 4,
                    type_embedding=False,
                    ) -> None:
        
        self.AminoAcid_Vocab = AminoAcid_Vocab
        self.token_size = len(self.AminoAcid_Vocab)
        self.residue_size = 21
        self.hidden_size = hidden_size
        self.pad_token_id = 0
        self.max_position_embeddings = max_position_embeddings
        self.type_residue_size = type_residue_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use__bias = use_bias
        self.num_hidden_layers = num_hidden_layers
        self.initializer_range = initializer_range
        self.type_embedding = type_embedding
