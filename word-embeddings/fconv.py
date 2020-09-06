# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
# Modified by Shashi Narayan (2018)

# Xin: code for NGTU and word2vec/glove with no padding
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from fairseq import utils
from fairseq.data import LanguagePairDataset
from fairseq.modules import BeamableMM, GradMultiply, LearnedPositionalEmbedding, LinearizedConvolution
from fairseq.vectordict import vector_dict

from . import FairseqEncoder, FairseqIncrementalDecoder, FairseqModel, register_model, register_model_architecture


@register_model('fconv')
class FConvModel(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.encoder.num_attention_layers = sum(layer is not None for layer in decoder.attention)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', default=0.1, type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-layers', type=str, metavar='EXPR',
                            help='encoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-layers', type=str, metavar='EXPR',
                            help='decoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='EXPR',
                            help='decoder attention [True, ...]')
        parser.add_argument('--share-input-output-embed', action='store_true',
                            help='share input and output embeddings (requires'
                                 ' --decoder-out-embed-dim and --decoder-embed-dim'
                                 ' to be equal)')

    @classmethod
    def build_model(cls, args, src_dict, dst_dict):
        """Build a new model instance."""
        encoder = FConvEncoder(
            src_dict,
            embed_dim=args.encoder_embed_dim,
            convolutions=eval(args.encoder_layers),
            dropout=args.dropout,
            max_positions=args.max_source_positions,
        )
        decoder = FConvDecoder(
            dst_dict,
            embed_dim=args.decoder_embed_dim,
            convolutions=eval(args.decoder_layers),
            out_embed_dim=args.decoder_out_embed_dim,
            attention=eval(args.decoder_attention),
            dropout=args.dropout,
            max_positions=args.max_target_positions,
            share_embed=args.share_input_output_embed
        )
        return FConvModel(encoder, decoder)


class FConvEncoder(FairseqEncoder):
    """Convolutional encoder"""
    def __init__(self, dictionary, embed_dim=512, max_positions=1024,
                 convolutions=((512, 3),) * 20, dropout=0.1):
        super().__init__(dictionary)
        embed_dim = vector_dict.embedding_dim
        convolutions=((vector_dict.embedding_dim, 3),) * 20
        self.dropout = dropout
        self.num_attention_layers = None
        self.embed_dim = embed_dim
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding.from_pretrained(torch.FloatTensor(vector_dict.embedding[:,:vector_dict.embedding_dim]), freeze=False)  # load pre-trained vector
        #self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        #self.embed_tokens.weight.data.copy_(torch.from_numpy(vector_dict.embedding))
        #self.embed_tokens.weight.requires_grad = True
        self.embed_positions = PositionalEmbedding(
            max_positions,
            embed_dim,
            padding_idx,
            left_pad=LanguagePairDataset.LEFT_PAD_SOURCE,
        )

        in_channels = convolutions[0][0]
        # Shashi
        self.fc1 = Linear(embed_dim+512, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        for (out_channels, kernel_size) in convolutions:
            self.projections.append(Linear(in_channels, out_channels)
                                    if in_channels != out_channels else None)
            self.convolutions.append(
                ConvTBC(in_channels, out_channels * 2, kernel_size,
                        dropout=dropout)
            )
            in_channels = out_channels
        self.fc2 = Linear(in_channels, embed_dim+512)
        self.lay_norm = nn.LayerNorm(embed_dim)  # layer nomalization in NGTU

    def forward(self, src_tokens, src_lengths, src_doctopic, src_wordtopics):
        # embed tokens and positions
        # print(self.embed_tokens(src_tokens), self.embed_positions(src_tokens), src_doctopic, src_wordtopics)

        # ''' 1)
        # src_doctopic: batchsize x 512
        # src_wordtopics: batchsize x wordcount x 512
        src_doctopic_ext = src_doctopic.unsqueeze(1) # batchsize x 1 x 512
        # print(src_doctopic_ext)
        src_wordtopics_doctopic = src_wordtopics * src_doctopic_ext # batchsize x wordcount x 512
        # print(src_wordtopics_doctopic)
        # ''' 

        ''' 2)
        # src_doctopic: batchsize x 512
        # src_wordtopics: batchsize x wordcount x 512
        src_doctopic_ext = src_doctopic.unsqueeze(1) # batchsize x 1 x 512
        # print(src_doctopic_ext)
        src_wordtopics_doctopic = src_wordtopics * src_doctopic_ext # batchsize x wordcount x 512
        # print(src_wordtopics_doctopic)
	# Normalize src_wordtopics_doctopic (April 29th)
        src_wordtopics_doctopic = F.normalize(src_wordtopics_doctopic, p=2, dim=2)
        '''
        
        
        x = self.embed_tokens(src_tokens) + self.embed_positions(src_tokens) # batchsize x wordcount x 512
        #word_embedding = torch.FloatTensor(vector_dict.get_embedding(src_tokens.cpu().numpy(), self.embed_dim)).to('cuda')
        #x = word_embedding + self.embed_positions(src_tokens) # batchsize x wordcount x 512
        # print(x)

        # Concat wordtopics*doctopic to (wordembedding+posembedding)
        x = torch.cat((x, src_wordtopics_doctopic), 2)
        # print(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        input_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # temporal convolutions
        for proj, conv in zip(self.projections, self.convolutions):
            residual = x if proj is None else proj(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            padding_l = (conv.kernel_size[0] - 1) // 2
            padding_r = conv.kernel_size[0] // 2
            x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
            x = conv(x)


            # NGTU BEGIN
            part_1, part_2 = x[:,:,:self.embed_dim],x[:,:,self.embed_dim:] 
            part_1 = torch.tanh(part_1)
            x = torch.cat([part_1,part_2],dim=2)
            x = F.glu(x, dim=2)
            x = self.lay_norm(x + residual)
            # NGTU END

            '''
            # original GLU BEGIN 
            x = F.glu(x, dim=2)
            x = (x + residual) * math.sqrt(0.5)
            # GLU END
            '''



        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # project back to size of embedding
        x = self.fc2(x)

        # scale gradients (this only affects backward, not forward)
        x = GradMultiply.apply(x, 1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        y = (x + input_embedding) * math.sqrt(0.5)

        # print(x,y)
        return x, y

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()

# original attension

class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim, bmm=None):
        super().__init__()
        embed_dim = vector_dict.embedding_dim
        # projects from output of convolution to embedding dimension
        self.in_projection = Linear(conv_channels, embed_dim+512)
        # projects from embedding dimension to convolution size
        self.out_projection = Linear(embed_dim+512, conv_channels)

        self.bmm = bmm if bmm is not None else torch.bmm

    def forward(self, x, target_embedding, encoder_out):
        residual = x

        # attention
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5) # d_i
        x = self.bmm(x, encoder_out[0]) # d_i*z_i

        # softmax over last dim
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x # a_i_j

        x = self.bmm(x, encoder_out[1]) # get c_i

        # scale attention output
        s = encoder_out[1].size(1)
        x = x * (s * math.sqrt(1.0 / s))

        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores

    def make_generation_fast_(self, beamable_mm_beam_size=None, **kwargs):
        """Replace torch.bmm with BeamableMM."""
        if beamable_mm_beam_size is not None:
            del self.bmm
            self.add_module('bmm', BeamableMM(beamable_mm_beam_size))

'''
# attension with general dot multiply
class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim, bmm=None):
        super().__init__()
        # projects from output of convolution to embedding dimension
        self.in_projection = Linear(conv_channels, embed_dim+embed_dim)
        # projects from embedding dimension to convolution size
        self.out_projection = Linear(embed_dim+embed_dim, conv_channels)

        self.attension_mul = Linear(embed_dim+embed_dim, embed_dim+embed_dim)

        self.bmm = bmm if bmm is not None else torch.bmm

    def forward(self, x, target_embedding, encoder_out):
        residual = x

        # attention
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5) # d_i
        #x = self.bmm(x, encoder_out[0]) # d_i*z_i
        x = self.bmm(self.attension_mul(x),encoder_out[0])

        # softmax over last dim
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x # a_i_j

        x = self.bmm(x, encoder_out[1]) # get c_i

        # scale attention output
        s = encoder_out[1].size(1)
        x = x * (s * math.sqrt(1.0 / s))

        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores

    def make_generation_fast_(self, beamable_mm_beam_size=None, **kwargs):
        """Replace torch.bmm with BeamableMM."""
        if beamable_mm_beam_size is not None:
            del self.bmm
            self.add_module('bmm', BeamableMM(beamable_mm_beam_size))
'''

'''
# attension with concat scores
class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim, bmm=None):
        super().__init__()
        # projects from output of convolution to embedding dimension
        self.in_projection = Linear(conv_channels, embed_dim+embed_dim)
        # projects from embedding dimension to convolution size
        self.out_projection = Linear(embed_dim+embed_dim, conv_channels)

        self.attension_mul = Linear(2*(embed_dim+embed_dim), 1)

        self.bmm = bmm if bmm is not None else torch.bmm

    def forward(self, x, target_embedding, encoder_out):
        residual = x

        # attention
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5) # d_i
        #x = self.bmm(x, encoder_out[0]) # d_i*z_i
        #x = self.bmm(self.attension_mul(x),encoder_out[0])
        enc_out = encoder_out[0].transpose(1,2)
        all_batch_x = []
        for i in range(x.shape[1]):
            single_batch = torch.cat([enc_out,torch.unsqueeze(x[:,1,:],1).repeat(1,enc_out.shape[1],1)],dim=2)
            #single_batch = self.attension_mul(single_batch)
            all_batch_x.append(single_batch.unsqueeze(2))
        all_batch_x = torch.cat(all_batch_x, dim=2)
        all_batch_x = self.attension_mul(all_batch_x).squeeze(3)
        #all_batch_x = torch.cat(all_batch_x, dim=2)
        x = all_batch_x.transpose(1,2)

        # softmax over last dim
        sz = x.size()
        x = F.softmax(x.contiguous().view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x # a_i_j

        x = self.bmm(x, encoder_out[1]) # get c_i

        # scale attention output
        s = encoder_out[1].size(1)
        x = x * (s * math.sqrt(1.0 / s))

        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores

    def make_generation_fast_(self, beamable_mm_beam_size=None, **kwargs):
        """Replace torch.bmm with BeamableMM."""
        if beamable_mm_beam_size is not None:
            del self.bmm
            self.add_module('bmm', BeamableMM(beamable_mm_beam_size))
'''

'''
# attension with  Gumbel-Softmax Trick and Re-parameterization Trick
class AttentionLayer(nn.Module):
    
    def __init__(self, conv_channels, embed_dim, bmm=None):
        super().__init__()
        # projects from output of convolution to embedding dimension
        self.in_projection = Linear(conv_channels, embed_dim+embed_dim)
        # projects from embedding dimension to convolution size
        self.out_projection = Linear(embed_dim+embed_dim, conv_channels)

        self.bmm = bmm if bmm is not None else torch.bmm
        
    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).cuda()
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y


    def forward(self, x, target_embedding, encoder_out):
        residual = x

        # attention
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5) # d_i
        x = self.bmm(x, encoder_out[0]) # d_i*z_i

        # Gumbel Softmax with multi samples
        sz = x.size()
        # sample for 10 times
        all_samles = []
        sample_num = 10
        for i in range (sample_num):
            # Gumbel Softmax
            tmp = self.gumbel_softmax(x.view(sz[0] * sz[1], sz[2]), 0.8)
            all_samles.append(tmp)
        x = torch.sum(torch.cat([tmp.unsqueeze(2) for tmp in all_samles],dim=2)/sample_num,dim=2)
        x = x.view(sz)
        attn_scores = x # a_i_j

        x = self.bmm(x, encoder_out[1]) # get c_i

        # scale attention output
        s = encoder_out[1].size(1)
        x = x * (s * math.sqrt(1.0 / s))

        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores

    def make_generation_fast_(self, beamable_mm_beam_size=None, **kwargs):
        """Replace torch.bmm with BeamableMM."""
        if beamable_mm_beam_size is not None:
            del self.bmm
            self.add_module('bmm', BeamableMM(beamable_mm_beam_size))
'''

'''
# multihead attension
class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim, bmm=None):
        super().__init__()
        self.num_head = 4
        # projects from output of convolution to embedding dimension
        self.in_projection = Linear(conv_channels, embed_dim+embed_dim)
        # projects from embedding dimension to convolution size
        self.out_projection = Linear(self.num_head*(embed_dim+embed_dim), conv_channels)
        self.bmm = bmm if bmm is not None else torch.bmm
        self.multi_head_key = []
        self.multi_head_value = []
        self.multi_head_query = []
        for i in range(self.num_head):
            self.multi_head_key.append(Linear(embed_dim+embed_dim, embed_dim+embed_dim).to('cuda'))
            self.multi_head_value.append(Linear(embed_dim+embed_dim, embed_dim+embed_dim).to('cuda'))
            self.multi_head_query.append(Linear(embed_dim+embed_dim, embed_dim+embed_dim).to('cuda'))
        #self.multi_head_fusion = Linear(self.num_head*(embed_dim+embed_dim), embed_dim+embed_dim)
    def forward(self, x, target_embedding, encoder_out):
        residual = x

        # attention
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5) # d_i

        # get new query and key
        all_query_result = []
        for i in range(self.num_head):
            
            # get new query and key
            new_query = self.multi_head_query[i](x)
            new_key = self.multi_head_key[i](encoder_out[0].transpose(1,2))
            new_key = new_key.transpose(1,2)
            new_queried = self.bmm(new_query, new_key)

            # softmax over last dim
            sz = new_queried.size()
            new_queried = F.softmax(new_queried.view(sz[0] * sz[1], sz[2]), dim=1)
            new_queried = new_queried.view(sz)
            attn_scores = new_queried # a_i_j

            # get new values
            new_value = self.multi_head_value[i](encoder_out[1])
            
            query_result = self.bmm(new_queried, new_value) # get c_i

            # scale attention output
            s = encoder_out[1].size(1)
            query_result = query_result * (s * math.sqrt(1.0 / s))
            all_query_result.append(query_result)
        all_query_result = torch.cat(all_query_result,dim=2)
        # project back
        x = (self.out_projection(all_query_result) + residual) * math.sqrt(0.5)
        return x, attn_scores

    def make_generation_fast_(self, beamable_mm_beam_size=None, **kwargs):
        """Replace torch.bmm with BeamableMM."""
        if beamable_mm_beam_size is not None:
            del self.bmm
            self.add_module('bmm', BeamableMM(beamable_mm_beam_size))
'''

'''
# addicative attension
class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim, bmm=None):
        super().__init__()
        # projects from output of convolution to embedding dimension
        self.in_projection = Linear(conv_channels, embed_dim+embed_dim)
        # projects from embedding dimension to convolution size
        self.out_projection = Linear(embed_dim+embed_dim, conv_channels)

        self.attension_add_query = Linear(embed_dim+embed_dim, embed_dim/2)
        self.attension_add_key = Linear(embed_dim+embed_dim, embed_dim/2)
        self.attension_add_score = Linear(embed_dim/2, 1)
        

        self.bmm = bmm if bmm is not None else torch.bmm

    def forward(self, x, target_embedding, encoder_out):
        residual = x

        # attention
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5) # d_i
        x = self.attension_add_query(x)
        #x = self.bmm(x, encoder_out[0]) # d_i*z_i
        #x = self.bmm(self.attension_mul(x),encoder_out[0])
        enc_out = encoder_out[0].transpose(1,2)
        enc_out = self.attension_add_key(enc_out)
        all_batch_x = []
        for i in range(x.shape[1]):
            single_batch = enc_out + torch.unsqueeze(x[:,1,:],1).repeat(1,enc_out.shape[1],1)
            #single_batch = self.attension_mul(single_batch)
            all_batch_x.append(single_batch.unsqueeze(2))
        all_batch_x = torch.cat(all_batch_x, dim=2)
        all_batch_x = torch.tanh(all_batch_x)
        all_batch_x = self.attension_add_score(all_batch_x).squeeze(3)
        #all_batch_x = torch.cat(all_batch_x, dim=2)
        x = all_batch_x.transpose(1,2)

        # softmax over last dim
        sz = x.size()
        x = F.softmax(x.contiguous().view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x # a_i_j

        x = self.bmm(x, encoder_out[1]) # get c_i

        # scale attention output
        s = encoder_out[1].size(1)
        x = x * (s * math.sqrt(1.0 / s))

        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores
'''


class FConvDecoder(FairseqIncrementalDecoder):
    """Convolutional decoder"""
    def __init__(self, dictionary, embed_dim=512, out_embed_dim=256,
                 max_positions=1024, convolutions=((512, 3),) * 20,
                 attention=True, dropout=0.1, share_embed=False):
        super().__init__(dictionary)
        embed_dim = vector_dict.embedding_dim
        self.embed_dim = embed_dim
        convolutions=((vector_dict.embedding_dim, 3),) * 20
        self.register_buffer('version', torch.Tensor([2]))
        self.dropout = dropout

        in_channels = convolutions[0][0]
        if isinstance(attention, bool):
            # expand True into [True, True, ...] and do the same with False
            attention = [attention] * len(convolutions)
        if not isinstance(attention, list) or len(attention) != len(convolutions):
            raise ValueError('Attention is expected to be a list of booleans of '
                             'length equal to the number of layers.')

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        self.embed_positions = PositionalEmbedding(
            max_positions,
            embed_dim,
            padding_idx,
            left_pad=LanguagePairDataset.LEFT_PAD_TARGET,
        )
        
        self.fc1 = Linear(embed_dim+512, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.lay_norm = nn.LayerNorm(embed_dim)
        for i, (out_channels, kernel_size) in enumerate(convolutions):
            self.projections.append(Linear(in_channels, out_channels)
                                    if in_channels != out_channels else None)
            self.convolutions.append(
                LinearizedConv1d(in_channels, out_channels * 2, kernel_size,
                                 padding=(kernel_size - 1), dropout=dropout)
            )
            self.attention.append(AttentionLayer(out_channels, embed_dim)
                                  if attention[i] else None)
            in_channels = out_channels
        self.fc2 = Linear(in_channels, out_embed_dim)
        if share_embed:
            assert out_embed_dim == embed_dim, \
                "Shared embed weights implies same dimensions " \
                " out_embed_dim={} vs embed_dim={}".format(out_embed_dim, embed_dim)
            self.fc3 = nn.Linear(out_embed_dim, num_embeddings)
            self.fc3.weight = self.embed_tokens.weight
        else:
            self.fc3 = Linear(out_embed_dim, num_embeddings, dropout=dropout)

    def forward(self, prev_output_tokens, encoder_out, src_doctopic, incremental_state=None):
        # split and transpose encoder outputs
        encoder_a, encoder_b = self._split_encoder_out(encoder_out, incremental_state)
        # print(encoder_a.size(), encoder_b.size())
        
        # embed tokens and combine with positional embeddings
        x = self._embed_tokens(prev_output_tokens, incremental_state)
        x += self.embed_positions(prev_output_tokens, incremental_state)
        # print(x.size())

        # Add doctopic vector in the decoder
        # src_doctopic: batchsize x 512
        src_doctopic_ext = src_doctopic.unsqueeze(1) # batchsize x 1 x 512
        # print(src_doctopic_ext.size())
        src_doctopic_ext = src_doctopic_ext.repeat(1, x.size()[1], 1)
        # print(src_doctopic_ext.size())

        # Concat doctopic to (wordembedding+posembedding)
        x = torch.cat((x, src_doctopic_ext), 2)
        # print(x.size())
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        target_embedding = x
        # print("Before FC1 ", x.size())

        # project to size of convolution
        x = self.fc1(x)
        # print("FC1 ", x.size())
        
        # B x T x C -> T x B x C
        x = self._transpose_if_training(x, incremental_state)
        
        # temporal convolutions
        avg_attn_scores = None
        num_attn_layers = len(self.attention)
        for proj, conv, attention in zip(self.projections, self.convolutions, self.attention):
            residual = x if proj is None else proj(x)

            x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = conv(x, incremental_state)

            '''
            # original GLU BEGIN
            x = F.glu(x, dim=2)
            # GLU END
            '''


            # NGTU BEGIN
            part_1, part_2 = x[:,:,:self.embed_dim],x[:,:,self.embed_dim:] 
            part_1 = torch.tanh(part_1)
            x = torch.cat([part_1,part_2],dim=2)
            x = F.glu(x, dim=2)
            # NGTU END


            # attention
            if attention is not None:
                x = self._transpose_if_training(x, incremental_state)
                # print(x.size())
                
                x, attn_scores = attention(x, target_embedding, (encoder_a, encoder_b))
                attn_scores = attn_scores / num_attn_layers
                if avg_attn_scores is None:
                    avg_attn_scores = attn_scores
                else:
                    avg_attn_scores.add_(attn_scores)

                x = self._transpose_if_training(x, incremental_state)

            '''
            # original GLU BEGIN
            # residual
            x = (x + residual) * math.sqrt(0.5)
            # GLU END
            '''


            # NGTU BEGIN
            x = self.lay_norm(x + residual)
            # NGTU END

        # T x B x C -> B x T x C
        x = self._transpose_if_training(x, incremental_state)
        # print(x.size())
        
        # project back to size of vocabulary
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)

        return x, avg_attn_scores

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if state_dict.get('decoder.version', torch.Tensor([1]))[0] < 2:
            # old models use incorrect weight norm dimension
            for i, conv in enumerate(self.convolutions):
                # reconfigure weight norm
                nn.utils.remove_weight_norm(conv)
                self.convolutions[i] = nn.utils.weight_norm(conv, dim=0)
            state_dict['decoder.version'] = torch.Tensor([1])
        return state_dict

    def _embed_tokens(self, tokens, incremental_state):
        if incremental_state is not None:
            # keep only the last token for incremental forward pass
            tokens = tokens[:, -1:]
        return self.embed_tokens(tokens)

    def _split_encoder_out(self, encoder_out, incremental_state):
        """Split and transpose encoder outputs.

        This is cached when doing incremental inference.
        """
        cached_result = utils.get_incremental_state(self, incremental_state, 'encoder_out')
        if cached_result is not None:
            return cached_result

        # transpose only once to speed up attention layers
        encoder_a, encoder_b = encoder_out
        encoder_a = encoder_a.transpose(1, 2).contiguous()
        result = (encoder_a, encoder_b)

        if incremental_state is not None:
            utils.set_incremental_state(self, incremental_state, 'encoder_out', result)
        return result

    def _transpose_if_training(self, x, incremental_state):
        if incremental_state is None:
            x = x.transpose(0, 1)
        return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, 0.1)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad):
    m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad)
    m.weight.data.normal_(0, 0.1)
    return m


def Linear(in_features, out_features, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def LinearizedConv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer optimized for decoding"""
    m = LinearizedConvolution(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m, dim=2)


def ConvTBC(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer"""
    from fairseq.modules import ConvTBC
    m = ConvTBC(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m, dim=2)


@register_model_architecture('fconv', 'fconv')
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', '[({}, 3)] * 20'.format(512))
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', '[({}, 3)] * 20'.format(512))
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')
    args.share_input_output_embed = getattr(args, 'share_input_output_embed', False)

@register_model_architecture('fconv', 'fconv_newsroom')
def fconv_newsroom(args):
    base_architecture(args)
    args.encoder_embed_dim = 256
    args.encoder_layers = '[(256, 3)] * 20'
    args.decoder_embed_dim = 256
    args.decoder_layers = '[(256, 3)] * 20'
    args.decoder_out_embed_dim = 256

@register_model_architecture('fconv', 'fconv_iwslt_de_en')
def fconv_iwslt_de_en(args):
    base_architecture(args)
    args.encoder_embed_dim = 256
    args.encoder_layers = '[(256, 3)] * 4'
    args.decoder_embed_dim = 256
    args.decoder_layers = '[(256, 3)] * 3'
    args.decoder_out_embed_dim = 256


@register_model_architecture('fconv', 'fconv_wmt_en_ro')
def fconv_wmt_en_ro(args):
    base_architecture(args)
    args.encoder_embed_dim = 512
    args.encoder_layers = '[({}, 3)] * 20'.format(512)
    args.decoder_embed_dim = 512
    args.decoder_layers = '[({}, 3)] * 20'.format(512)
    args.decoder_out_embed_dim = 512


@register_model_architecture('fconv', 'fconv_wmt_en_de')
def fconv_wmt_en_de(args):
    base_architecture(args)
    convs = '[({}, 3)] * 9'.format(512)       # first 9 layers have 512 units
    convs += ' + [(1024, 3)] * 4'  # next 4 layers have 1024 units
    convs += ' + [(2048, 1)] * 2'  # final 2 layers use 1x1 convolutions
    args.encoder_embed_dim = 768
    args.encoder_layers = convs
    args.decoder_embed_dim = 768
    args.decoder_layers = convs
    args.decoder_out_embed_dim = 512


@register_model_architecture('fconv', 'fconv_wmt_en_fr')
def fconv_wmt_en_fr(args):
    base_architecture(args)
    convs = '[({}, 3)] * 6'.format(512)       # first 6 layers have 512 units
    convs += ' + [(768, 3)] * 4'   # next 4 layers have 768 units
    convs += ' + [(1024, 3)] * 3'  # next 3 layers have 1024 units
    convs += ' + [(2048, 1)] * 1'  # next 1 layer uses 1x1 convolutions
    convs += ' + [(4096, 1)] * 1'  # final 1 layer uses 1x1 convolutions
    args.encoder_embed_dim = 768
    args.encoder_layers = convs
    args.decoder_embed_dim = 768
    args.decoder_layers = convs
    args.decoder_out_embed_dim = 512
