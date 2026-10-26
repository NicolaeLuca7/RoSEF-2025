import torch
from torch import nn
import math

import math

class ParametersEncoder(nn.Module):
    def __init__(self,input_size,hid_dim,embed_dim,dropout):
        super().__init__()

        self.embed_dim = embed_dim
        self.sequential = nn.Sequential(
            nn.Linear(input_size,hid_dim),
            nn.Dropout(dropout),
            nn.Linear(hid_dim,hid_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim,embed_dim),
        )
        
    def forward(self,x):
        x = self.sequential(x) * 0.1
        return x

class WaveEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim,embed_dim)

        scale_factor = 0.1
    
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        self.linear.weight.data.mul_(scale_factor)
        
        self.linear.bias.data.zero_()
    
    def forward(self,x):
        x = self.linear(x)
        return x

class WaveDecoder(nn.Module):
    def __init__(self, embed_dim, hid_dim, dropout):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Linear(embed_dim,1),
        )
    
    def forward(self,x):
        x = self.sequential(x)
        return x

class PositionalEmbeddings(nn.Module):
    def __init__(self, embed_dim,max_seq_len, dropout, scale_factor=0.1):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = scale_factor

    def forward(self, x,sos_token):
        T = x.shape[-2]
        
        positions = torch.arange(T, device=x.device)
        pos_embeds = self.pos_embedding(positions).unsqueeze(0)
        pos_embeds *= self.scale

        context_expanded = sos_token.unsqueeze(1).expand(-1, T, -1)
        pos_embeds = pos_embeds + context_expanded

        x = x + pos_embeds
        return self.dropout(x)


class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_dim, heads):
        super().__init__()

        self.embed_dim = embed_dim
        self.heads = heads

        if embed_dim % heads != 0:
            raise ValueError("embed_dim not divisible by heads")
        self.dk = embed_dim // heads
        self.d_k_sqrt = math.sqrt(self.dk)
        
        self.qproj = nn.Linear(embed_dim, embed_dim)
        self.kproj = nn.Linear(embed_dim, embed_dim)
        self.vproj = nn.Linear(embed_dim, embed_dim)
        
        self.softmax = nn.Softmax(dim=-1)


    def split_into_heads(self,x):
        batch_size, seq_length, embed_dim = x.shape
        x = x.view(batch_size, seq_length, self.heads, self.dk)
        return x.transpose(1, 2).contiguous()
    
    def combine_head(self,x):
        batch_size, heads, seq_length, dk = x.shape
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size,seq_length,heads*dk)
        return x

    def attention(self,q,k,v,mask=None,padding_mask=None):
        scores = torch.matmul(q, torch.transpose(k, -2, -1))
        scores = scores / self.d_k_sqrt
        
        if mask is not None:
            scores = scores + mask

        if padding_mask is not None:
            scores = scores + padding_mask

        attn_weights = self.softmax(scores)
        output = torch.matmul(attn_weights,v)

        return output
    
    def forward(self,x,mask=None,padding_mask=None):
        q = self.qproj(x)
        k = self.kproj(x)
        v = self.vproj(x) 

        q = self.split_into_heads(q)
        k = self.split_into_heads(k)
        v = self.split_into_heads(v)

        x = self.attention(q,k,v,mask,padding_mask)
        x = self.combine_head(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, embed_dim,ff_dim):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Linear(embed_dim,ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim,embed_dim),
        )

    def forward(self,x):
        x = self.sequential(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self,embed_dim,masked_heads,ff_dim,dropout):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.multi_head = MaskedSelfAttention(embed_dim = embed_dim, heads = masked_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim = embed_dim,ff_dim = ff_dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self,x,mask,padding_mask):
        x = self.layer_norm1(x)

        masked_att = self.dropout1(self.multi_head(x,mask,padding_mask))
        x = x + masked_att

        x = self.layer_norm2(x)
        ff_x = self.dropout2(self.feed_forward(x))
        x = x + ff_x

        return x


class Decoder(nn.Module):
    def __init__(self,embed_dim,max_seq_len,dropout,decoders_num, masked_heads,ff_dim):
        super().__init__()

        self.positional_embeddings = PositionalEmbeddings(embed_dim = embed_dim,max_seq_len = max_seq_len,dropout = dropout)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(embed_dim = embed_dim,masked_heads = masked_heads,ff_dim = ff_dim,dropout=dropout) for i in range(decoders_num)])
        self.linear1 = nn.Linear(embed_dim,embed_dim)

    def forward(self,x,mask,padding_mask,sos_token):
        x = self.positional_embeddings(x,sos_token)
        for block in self.decoder_blocks:
            x = block(x,mask,padding_mask)
        x = self.linear1(x)

        return x


class WaveformPredictor(nn.Module):
    def __init__(self,parameters):
        super().__init__()

        self.device = parameters['device']
        self.pad_token = parameters['pad_token']
        self.embed_dim = parameters['embed_dim']

        self.parameters_encoder = ParametersEncoder(input_size = parameters['input_size'], hid_dim = parameters['encoder_hid_dim'], embed_dim = parameters['embed_dim'], dropout = parameters['encoder_dropout'])
        self.wave_encoder = WaveEncoder(parameters['wave_dim'],parameters['embed_dim'])
        self.decoder = Decoder(embed_dim = parameters['embed_dim'],max_seq_len = parameters['max_seq_len'], dropout = parameters['decoder_dropout'], decoders_num = parameters['decoders_num'], masked_heads = parameters['masked_heads'], ff_dim = parameters['ff_hid_dim'])
        self.wave_decoder = WaveDecoder(embed_dim = parameters['embed_dim'], hid_dim = parameters['decoder_hid_dim'], dropout = parameters['wave_decoder_dropout'])
    
    def get_square_subsequent_mask(self, size: int):
        mask = torch.triu(torch.ones(size, size, device=self.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
        return mask
    
    def get_square_subsequent_mask(self, size: int):
        mask = torch.ones(size, size, device=self.device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def decode(self,x,pad_mask,sos_token):
        self.padding_mask = pad_mask.float().unsqueeze(1).unsqueeze(1)
        self.padding_mask = self.padding_mask.masked_fill(self.padding_mask == 0, float('-inf')).masked_fill(self.padding_mask == 1, float(0)).to(x.device)
       
        self.mask = self.get_square_subsequent_mask(x.size(-2))
        x = self.decoder(x,self.mask,self.padding_mask,sos_token)

        return x

    def forward(self,x_param,x_wave,pad_mask):
        sos_token = self.parameters_encoder(x_param)
        
        x_wave = self.wave_encoder(x_wave)

        x_wave = torch.cat([sos_token.unsqueeze(dim=1),x_wave], dim=1)

        wave_output = self.decode(x_wave,pad_mask,sos_token)
        wave_output = self.wave_decoder(wave_output)
        

        return wave_output
