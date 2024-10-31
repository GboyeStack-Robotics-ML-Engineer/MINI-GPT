import torch
import MultiheadAttention
from MultiheadAttention import MULTIHEADSELFATTN
import numpy as np

class DECODERLAYER(torch.nn.Module):
    
    def __init__(self,NUM_HEAD,EMBED_DIM,use_casual,MLP_ratio, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.q_MASKED_proj=torch.nn.Linear(EMBED_DIM,EMBED_DIM)
        self.k_MASKED_proj=torch.nn.Linear(EMBED_DIM,EMBED_DIM)
        self.v_MASKED_proj=torch.nn.Linear(EMBED_DIM,EMBED_DIM)
        
        self.q_AFTER_MASKED_proj=torch.nn.Linear(EMBED_DIM,EMBED_DIM)
        
        self.NUM_HEAD=NUM_HEAD
        self.EMBED_DIM=EMBED_DIM
        
        self.MHA_MASK=MULTIHEADSELFATTN(NUM_HEAD=NUM_HEAD,use_casual=use_casual,attention_type=None)
        
        self.norm1 = torch.nn.LayerNorm(self.EMBED_DIM)
        
        self.MHA_CROSS=MULTIHEADSELFATTN(NUM_HEAD=NUM_HEAD,use_casual=False,attention_type=None)
        
        self.norm2=torch.nn.LayerNorm(self.EMBED_DIM)
        
        self.MLP_ratio=MLP_ratio
        
        self.mlp_point_wise_attn= torch.nn.Sequential(
            torch.nn.Linear(self.EMBED_DIM, self.MLP_ratio * self.EMBED_DIM),
            torch.nn.GELU(),
            torch.nn.Linear(self.MLP_ratio * self.EMBED_DIM, self.EMBED_DIM)
        )
        
        self.norm3=torch.nn.LayerNorm(self.EMBED_DIM)
    def positional_encoding(self,tokens,plot=False):
        encodings=torch.ones_like(tokens[0,:,:,])
        for i in range(encodings.size()[0]):
            for j in range(encodings.size()[1]):
                encodings[i,j]=np.sin(i / (10000 ** (j / self.EMBED_DIM)) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / self.EMBED_DIM))))
        
        return encodings
    
    def forward(self,inputs,k_from_encoder,v_from_encoder):
        
        positional_encodings=self.positional_encoding(inputs).repeat(inputs.shape[0],1,1) #Function to generate positional encodings will be implemeted later
        
        x=inputs+positional_encodings

        assert 2<=x.dim()<=3,f'Expected the input to be 3D or 2D tensors, but got {x.dim()}D'
        if x.dim()==2:
            x=x.unsqueeze(0)
            
        self.BSZ,self.SEQ_LEN,self.EMBED_DIM=x.shape
        
        assert self.EMBED_DIM==x.shape[-1], f'MULTIHEADSELFATTN.forward() method expected the EMBED_DIM:{self.EMBED_DIM}  to match the input dimension. Got Input dim:{x.shape[-1]}, EMBED_DIM:{self.EMBED_DIM}'
        
        Q=self.q_MASKED_proj(x).reshape(self.BSZ,self.SEQ_LEN,self.NUM_HEAD,self.EMBED_DIM//self.NUM_HEAD)
        K=self.k_MASKED_proj(x).reshape(self.BSZ,self.SEQ_LEN,self.NUM_HEAD,self.EMBED_DIM//self.NUM_HEAD)
        V=self.v_MASKED_proj(x).reshape(self.BSZ,self.SEQ_LEN,self.NUM_HEAD,self.EMBED_DIM//self.NUM_HEAD)
        
        contextualized_embeddings=self.MHA_MASK(Q,K,V,return_attention=False).to(x.device)

        masked_attention_layer_output=self.norm1(contextualized_embeddings+x)
        
        Q_to_decoder=self.q_AFTER_MASKED_proj(masked_attention_layer_output).reshape(self.BSZ,self.SEQ_LEN,self.NUM_HEAD,self.EMBED_DIM//self.NUM_HEAD)

        contextualized_embeddings=self.MHA_MASK(Q_to_decoder,k_from_encoder,v_from_encoder,return_attention=False).to(masked_attention_layer_output.device)

        attention_layer_output=self.norm2(contextualized_embeddings+masked_attention_layer_output)
        
        mlp_layer_output=self.mlp_point_wise_attn(attention_layer_output)
        
        resulting_layer_output=self.norm3(attention_layer_output+mlp_layer_output)
        
        return resulting_layer_output

# BSZ,SEQ_LEN,VOCAB_SIZE=2,3,5
# NUM_HEAD,EMBED_DIM=4,512

# embedding=torch.randn(size=(BSZ,SEQ_LEN,EMBED_DIM))
# encoder=ENCODERLAYER(NUM_HEAD=4,EMBED_DIM=512,use_casual=True,MLP_ratio=2)
# out=encoder(embedding)
# print(out[0,:,:].detach().numpy())
# import seaborn as sns
# import matplotlib.pyplot as plt
# fig=plt.figure(1,figsize=(20,20))
# ax=fig.add_subplot(1,1,1)

# sns.heatmap(out[0,:,:].detach().numpy(),ax=ax)
# plt.show()
