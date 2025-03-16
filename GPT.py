import torch
from ENCODER import ENCODERLAYER
from DECODER import DECODERLAYER
from collections import OrderedDict
import copy
torch.use_deterministic_algorithms(mode=True,warn_only=True)
import numpy as np
class MODEL(torch.nn.Module):
    def __init__(self,NUM_ENCODER_LAYER,
                      NUM_DECODER_LAYER,
                      NUM_ENCODER_HEAD,
                      NUM_DECODER_HEAD,
                      ENCODER_EMBED_DIM,
                      DECODER_EMBED_DIM,
                      ENCODER_MLP_ratio,
                      DECODER_MLP_ratio,
                      VOCAB_SIZE,
                *args, **kwargs):
        
    
        super().__init__(*args, **kwargs)
        
        self.NUM_ENCODER_LAYER=NUM_ENCODER_LAYER 
        self.NUM_DECODER_LAYER=NUM_DECODER_LAYER
        self.NUM_ENCODER_HEAD=NUM_ENCODER_HEAD
        self.NUM_DECODER_HEAD=NUM_DECODER_HEAD
        self.ENCODER_EMBED_DIM=ENCODER_EMBED_DIM
        self.DECODER_EMBED_DIM=DECODER_EMBED_DIM
        self.ENCODER_MLP_ratio=ENCODER_MLP_ratio
        self.DECODER_MLP_ratio=DECODER_MLP_ratio
        self.VOCAB_SIZE=VOCAB_SIZE
        

        
        self.input_embedings=torch.nn.Embedding(embedding_dim=self.ENCODER_EMBED_DIM,num_embeddings=self.VOCAB_SIZE)
        
        self.target_embedings=torch.nn.Embedding(embedding_dim=self.DECODER_EMBED_DIM,num_embeddings=self.VOCAB_SIZE)
        
        encoder_module=ENCODERLAYER(NUM_HEAD=self.NUM_ENCODER_HEAD,
                                    EMBED_DIM=self.ENCODER_EMBED_DIM,
                                    use_casual=False,
                                    MLP_ratio=self.ENCODER_MLP_ratio)
        

        decoder_module=DECODERLAYER(NUM_HEAD=self.NUM_DECODER_HEAD,
                                    EMBED_DIM=self.DECODER_EMBED_DIM,
                                    use_casual=True,
                                    MLP_ratio=self.DECODER_MLP_ratio)
        
        self.encoderblock=torch.nn.ModuleList([copy.deepcopy(encoder_module) for i in range(self.NUM_ENCODER_LAYER)]) #solution to error was: ModuleList([copy.deepcopy(module) for i in range(N)])
        
        # self.k_proj=torch.nn.Linear(self.ENCODER_EMBED_DIM,self.ENCODER_EMBED_DIM)
        # self.v_proj=torch.nn.Linear(self.ENCODER_EMBED_DIM,self.ENCODER_EMBED_DIM)
    
        self.decoderblock=torch.nn.ModuleList([copy.deepcopy(decoder_module) for i in range(self.NUM_DECODER_LAYER)])
        
        self.classifier_head=torch.nn.Linear(self.DECODER_EMBED_DIM,VOCAB_SIZE)
            
    
    def positional_encoding(self,tokens,embed_dim):
        encodings=torch.ones_like(tokens[0,:,:,])
        for i in range(encodings.size()[0]):
            for j in range(encodings.size()[1]):
                encodings[i,j]=np.sin(i / (10000 ** (j / embed_dim))) if i % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / embed_dim)))
    
        return encodings
    
    def create_padding_mask(self,seq):
        # print(seq)
        seq=seq.squeeze(1)
        BSZ,SEQ_LEN=seq.shape
        
        
        padding_mask=torch.eq(seq,0)
        
        return padding_mask.view(BSZ,1,1,SEQ_LEN)
        # pass
    
    def create_future_mask(self,seq):
        seq=seq.squeeze(1)
        BSZ,SEQ_LEN=seq.shape
        
        mask=torch.triu(torch.ones((1,1,SEQ_LEN,SEQ_LEN)),diagonal=1)==0
        
        return mask
    
    def create_masks(self,src,tgt,input_padding_mask,target_padding_mask):
        
        src_padding_mask=self.create_padding_mask(input_padding_mask).to(src.device)
        
        tgt_padding_mask=self.create_padding_mask(target_padding_mask).to(tgt.device)
        
        tgt_future_mask=self.create_future_mask(tgt).to(tgt.device)
        
        # print(src_padding_mask.shape,tgt_padding_mask.shape,tgt_future_mask.shape)
        
        tgt_mask=tgt_padding_mask & tgt_future_mask  
        
        return src_padding_mask,tgt_padding_mask,tgt_mask
        
    def forward(self,input_ids=None,label_ids=None,input_padding_mask=None,target_padding_mask=None,return_loss=True):
        
        # padding_mask=self.create_padding_mask(padding_mask)
        
        # future_mask=self.create_future_mask(input_ids)
        
        src_pad_mask,tgt_pad_mask,tgt_mask=self.create_masks(src=input_ids,tgt=label_ids,input_padding_mask=input_padding_mask,target_padding_mask=target_padding_mask)
        
        # print(src_pad_mask)
        # print(tgt_pad_mask)
        # print(tgt_mask)
        
        # print(padding_mask)
        # print(future_mask)
        
        input_embeddings=self.input_embedings(input_ids) 
        
        encoder_positional_encodings=self.positional_encoding(input_embeddings,embed_dim=self.ENCODER_EMBED_DIM)
        
        positioned_input_embeddings=input_embeddings+encoder_positional_encodings
        
        
        target_embeddings=self.target_embedings(label_ids) 
        
        target_positional_encodings=self.positional_encoding(target_embeddings,embed_dim=self.DECODER_EMBED_DIM)
        
        positioned_target_embeddings=target_embeddings + target_positional_encodings
        
        
        x=input_embeddings.squeeze(1)
        
        y=target_embeddings.squeeze(1)
    
        # self.BSZ,self.SEQ_LEN,self.EMBED_DIM=x.shape
        
        for index,encoder_layer in enumerate(self.encoderblock):
            
            x=encoder_layer(x,src_mask=src_pad_mask)
        
        
        # K_from_encoder=self.k_proj(x).reshape(self.BSZ,self.SEQ_LEN,self.NUM_ENCODER_HEAD,self.EMBED_DIM//self.NUM_ENCODER_HEAD)
        # V_from_encoder=self.v_proj(x).reshape(self.BSZ,self.SEQ_LEN,self.NUM_ENCODER_HEAD,self.EMBED_DIM//self.NUM_ENCODER_HEAD)
        
        for decoder_layer in self.decoderblock:
            
            y=decoder_layer(target=y,encoder_output=x,src_mask=src_pad_mask,tgt_mask=tgt_mask,target_padding_mask=tgt_pad_mask)
        
        prediction_logits=self.classifier_head(y)
        
        # print(prediction_logits.shape)
        
        token_probabilities=torch.nn.Softmax(dim=-1)(prediction_logits)
        
        if return_loss:
            one_hot_labels=torch.nn.functional.one_hot(label_ids,num_classes=self.VOCAB_SIZE).type(dtype=torch.float)
            
            loss=torch.nn.functional.cross_entropy(input=prediction_logits,target=one_hot_labels.squeeze(1))
            
            return {'loss':loss,'predictions':token_probabilities}
        
        return token_probabilities





# BSZ,SEQ_LEN,VOCAB_SIZE=2,3,5
# NUM_HEAD,EMBED_DIM=4,64

# model=MODEL(NUM_ENCODER_LAYER=6,
#             NUM_DECODER_LAYER=6,
#             NUM_ENCODER_HEAD=NUM_HEAD,
#             NUM_DECODER_HEAD=NUM_HEAD,
#             ENCODER_EMBED_DIM=1024,
#             DECODER_EMBED_DIM=1024,
#             ENCODER_MLP_ratio=4,
#             DECODER_MLP_ratio=4,
#             VOCAB_SIZE=VOCAB_SIZE)
# # print(model)
# # print(f'Total Number of model paramters: {sum([p.numel() for p in model.parameters(recurse=True)])}')
# # # print(f'Total Number of  Trainable model paramters: {sum([p.numel() for p in model.parameters(recurse=True) if p.requires_grad==True])}')
# #print([name for name, param in model.state_dict().items()])
# input_ids=torch.randint(low=0,high=VOCAB_SIZE,size=(BSZ,SEQ_LEN))
# target_ids=torch.randint(low=0,high=VOCAB_SIZE,size=(BSZ,SEQ_LEN))
# input_padding_mask=torch.randint(low=0,high=2,size=(BSZ,SEQ_LEN))
# tgt_padding_mask=torch.randint(low=0,high=2,size=(BSZ,SEQ_LEN))

# output=model(input_ids=input_ids,label_ids=target_ids,return_loss=True,input_padding_mask=input_padding_mask,target_padding_mask=tgt_padding_mask)
# print(output['loss'])


# import tqdm
# from tqdm import tqdm as bar1
# import time
# i=0
# for y in bar1(range(100),colour='green',dynamic_ncols=True,desc='Training',bar_format='{desc}|{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):#,bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
#     # for i in bar1(range(100),leave=False):
#     #     i+=1
#     #     time.sleep(0.1)
#     i+=1
#     # bar1.write(f'Current epoch:{y}')
    
    