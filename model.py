import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(q, k, v, mask=None, adjoin_matrix=None):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1)) 

    dk = k.shape[-1]
    scaled_attention_logits = matmul_qk / math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += mask * -1e9

    if adjoin_matrix is not None:
        scaled_attention_logits += adjoin_matrix

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)  

    return output, attention_weights



class MultiHeadAttention(nn.Module):
  def __init__(self,d_model,num_heads):
    super().__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    self.depth = d_model // self.num_heads

    self.wq = nn.Linear(d_model,d_model)
    self.wk = nn.Linear(d_model,d_model)
    self.wv = nn.Linear(d_model,d_model)
    self.dense = nn.Linear(d_model,d_model)

  def forward(self,v,k,q,mask,adjoin_matrix):
    batch_size = q.shape[0]

    q = self.wq(q)
    k = self.wk(k)
    v = self.wv(v)

    q = self.split_heads(q,batch_size)
    k = self.split_heads(k,batch_size)
    v = self.split_heads(v,batch_size)

    scaled_attention,attention_weights = scaled_dot_product_attention(q,k,v,mask,adjoin_matrix)
    scaled_attention = scaled_attention.permute(0,2,1,3)
    concat_attention = scaled_attention.reshape(batch_size,-1, self.d_model)
    output = self.dense(concat_attention)
    return output,attention_weights

  def split_heads(self,x,batch_size):
    x = torch.reshape(x,(batch_size,-1,self.num_heads,self.depth))
    x = x.permute(0,2,1,3)
    return x




class FeedForwardNetwork(nn.Module):
  def __init__(self,dff,d_model):
    super().__init__()
    self.fc1 = nn.Linear(d_model,dff)
    self.fc2 = nn.Linear(dff,d_model)

  def forward(self,x):
    x = F.gelu(self.fc1(x))
    x = self.fc2(x)
    return x


class EncoderLayer(nn.Module):
  def __init__(self,d_model,num_heads,dff,rate = 0.1):
    super().__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = FeedForwardNetwork(dff,d_model)
    self.layernorm1 = torch.nn.LayerNorm(d_model)
    self.layernorm2 = torch.nn.LayerNorm(d_model)

    self.dropout1 = torch.nn.Dropout(rate)
    self.dropout2 = torch.nn.Dropout(rate)

  def forward(self,x,training,mask, adjoin_matrix):
    attn_output , attention_weights = self.mha(x,x,x,mask,adjoin_matrix)
    attn_output = self.dropout1(attn_output)
    out1 = self.layernorm1(x + attn_output)
    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output)
    out2 = self.layernorm2(out1 + ffn_output)
    return out2,attention_weights
  



  



class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               maximum_position_encoding, rate=0.1):
        super().__init__()

        

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, rate) 
                                         for _ in range(num_layers)])
        self.dropout = nn.Dropout(rate)

    def forward(self, x, training, mask, adjoin_matrix):
        seq_len = x.shape[1]
        if adjoin_matrix is not None:
            adjoin_matrix = adjoin_matrix.unsqueeze(1)

        x = self.embedding(x)
        x *= math.sqrt(self.d_model)

        
        x = self.dropout(x)

        for layer in self.enc_layers:
            x, attention_weights = layer(x, training, mask, adjoin_matrix)

        return x, attention_weights
    
class CrossAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=8, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        out, weights = self.attn(
            query=query,
            key=key_value,
            value=key_value
        )

        out = self.norm(query + self.dropout(out))
        return out, weights
    


class SmilesCNNEncoder(nn.Module):
    def __init__(self, vocab_size=18, d_model=256, out_dim=256, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.conv3 = nn.Conv1d(d_model, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(d_model, 128, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(d_model, 128, kernel_size=7, padding=3)

        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Sequential(
            nn.Linear(128 * 3, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        # x shape: [batch, sequence_length]
        x = self.embedding(x)

        # Conv1d expects [batch, channels, sequence_length]
        x = x.transpose(1, 2)

        h3 = F.gelu(self.conv3(x))
        h5 = F.gelu(self.conv5(x))
        h7 = F.gelu(self.conv7(x))

        # Global max pooling
        h3 = torch.max(h3, dim=2).values
        h5 = torch.max(h5, dim=2).values
        h7 = torch.max(h7, dim=2).values

        x = torch.cat([h3, h5, h7], dim=1)
        x = self.dropout(x)

        return self.proj(x)



class PredictModel(nn.Module):
    def __init__(self, num_layers=6, d_model=256, dff=512, num_heads=8, vocab_size=18, dropout_rate=0.5):
        super().__init__()


        self.maccs_proj = nn.Sequential(
            nn.Linear(334, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(128)
        )

        self.aac_proj = nn.Sequential(
            nn.Linear(60, 64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(64)
        )

        self.payload_encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            input_vocab_size=vocab_size,
            maximum_position_encoding=200,
            rate=dropout_rate
        )

        self.linker_encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            input_vocab_size=vocab_size,
            maximum_position_encoding=200,
            rate=dropout_rate
            )
        self.payload_pool = nn.Linear(d_model, 1)
        self.linker_pool = nn.Linear(d_model, 1)


        self.payload_cnn = SmilesCNNEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            out_dim=256,
            dropout=dropout_rate
        )

        self.linker_cnn = SmilesCNNEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            out_dim=256,
            dropout=dropout_rate
        )

        self.payload_merge = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(256)
        )

        self.linker_merge = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(256)
        )


        self.heavy_transform= nn.Linear(2592, 1280)
        self.antigen_transform = nn.Linear(1280, 1280)

        self.cross_attn = CrossAttention(
            d_model=1280,
            num_heads=8
        )

        fusion_dim = 4546

        self.fusion_proj = nn.Linear(fusion_dim, fusion_dim)
        self.fusion_norm = nn.LayerNorm(fusion_dim)

        self.fusion_dropout = nn.Dropout(dropout_rate)
        self.dar_scale = nn.Parameter(torch.tensor(10.0))

        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim,512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, x1, x1maccs, x2, x2maccs, t1, t2, t3, aac1, aac2, aac3, t4, 
                adjoin_matrix1=None, mask1=None, adjoin_matrix2=None, mask2=None, training=False):
        
        x1_tokens = x1
        x2_tokens = x2

        
        x1, attention_weights1 = self.payload_encoder(
            x1,
            training=training,
            mask=mask1,
            adjoin_matrix=adjoin_matrix1
        )

        payload_weights = torch.softmax(
            self.payload_pool(x1), dim=1
        )
        self.payload_attention_weights = (
            payload_weights.detach().cpu()
        )
        x1 = (x1 * payload_weights).sum(dim=1)

        x1_cnn = self.payload_cnn(x1_tokens)
        x1 = self.payload_merge(torch.cat([x1, x1_cnn], dim=1))


        x2, attention_weights2 = self.linker_encoder(
            x2,
            training=training,
            mask=mask2,
            adjoin_matrix=adjoin_matrix2
        )

        linker_weights = torch.softmax(
            self.linker_pool(x2), dim=1
        )
        self.linker_attention_weights = linker_weights.detach().cpu()
        x2 = (x2 * linker_weights).sum(dim=1)

        x2_cnn = self.linker_cnn(x2_tokens)
        x2 = self.linker_merge(torch.cat([x2, x2_cnn], dim=1))

        if t4.dim() == 1:
            t4 = t4.unsqueeze(1)

        # project antibody and antigen
        ab_vec = self.heavy_transform(t1)   # [batch, 1280]
        ag_vec = self.antigen_transform(t3) # [batch, 1280]

        ab = ab_vec.unsqueeze(1)
        ag = ag_vec.unsqueeze(1)

        # cross-attention
        ab_to_ag, _ = self.cross_attn(ab, ag)
        ag_to_ab, _ = self.cross_attn(ag, ab)

        ab_to_ag = ab_to_ag.squeeze(1)
        ag_to_ab = ag_to_ab.squeeze(1)

        cross_out = torch.cat((ab_to_ag ,ag_to_ab),dim=1)

        extra_pad = torch.zeros(x1.shape[0], 31, device=x1.device)
        global_interaction_score= cross_out.mean(dim=1, keepdim=True)

        if t4.dim() == 1:
            t4 = t4.unsqueeze(1)

        t4 = (t4 / 8.0) * 100

        maccs_feat = self.maccs_proj(torch.cat([x1maccs, x2maccs], dim=1))
        aac_feat = self.aac_proj(torch.cat([aac1, aac2, aac3], dim=1))


        # final fusion
        x = torch.cat((
            x1,         # 256(payload)
            x2,         # 256(linker)
            t2,         # 1280(light chain)
            cross_out,  # 2560(bi direction attention)
            t4,         # 1(dar)
            global_interaction_score,  #1
            maccs_feat,
            aac_feat
        ),dim=1)
        
        fusion_residual = self.fusion_proj(x)
        fusion_residual = F.gelu(fusion_residual)
        fusion_residual = self.fusion_dropout(fusion_residual)

        x = self.fusion_norm(x + fusion_residual)

        x = self.mlp(x)
        x = x.squeeze(1)

        return x
