import torch
import torch.multiprocessing as mp 


class MULTIHEADSELFATTN(torch.nn.Module):
    
    def __init__(self,NUM_HEAD,attention_type,use_casual=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.NUM_HEAD=NUM_HEAD
        self.attention_type=attention_type
        self.use_casual_attn=use_casual
        self.Q,self.K,self.V=None,None,None
        self.BSZ,self.SEQ_LEN,self.EMBED_DIM,self.HEAD_DIM=None,None,None,None
        
    def forward(self,Q,K,V,return_attention=True,attention_mask=None,padding_mask=None):
        
        self.Q,self.K,self.V=Q,K,V
        
        self.BSZ,self.SEQ_LEN,self.NUM_HEAD,self.HEAD_DIM=Q.shape
        
        self.EMBED_DIM=self.HEAD_DIM*self.NUM_HEAD
        
        attention_scores=torch.softmax(self.Q@self.K.transpose(-2,-1)/torch.sqrt(input=torch.tensor(self.EMBED_DIM).type(dtype=torch.int64)),dim=-1)
        
        if (attention_mask is None):
            
            if (self.use_casual_attn):
        
                attention_mask=torch.nn.Transformer.generate_square_subsequent_mask(self.NUM_HEAD).unsqueeze(0).unsqueeze(0)
                    
                attention_mask[attention_mask==-torch.inf]=-1E15
            else:
                attention_mask=torch.zeros_like(torch.nn.Transformer.generate_square_subsequent_mask(self.NUM_HEAD).unsqueeze(0).unsqueeze(0))
                        
        else:
            assert attention_mask.dim()==2 , f'{Q.dim()-1}-d input expects a 2d attention mask, got {attention_mask.dim()}'
            
            attention_mask=attention_mask.unsqueeze(0).unsqueeze(0)
        
        if padding_mask is not None:
            
            pass

        attention_scores=attention_scores.to(attention_mask.device)
        
        attention_scores=attention_scores+attention_mask  
        
        layer_output=attention_scores@V.to(attention_scores.device)
                
        layer_output=layer_output.view(self.BSZ,self.SEQ_LEN,self.EMBED_DIM)
        
        if return_attention:
            return attention_scores,layer_output
        
        else:
            return layer_output
        
# attention,output=MULTIHEADSELFATTN(NUM_HEAD=4)(embedding)

# print(output)

    
    




    