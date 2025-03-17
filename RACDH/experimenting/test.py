import sys
import os
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
# from sentence_transformers import CrossEncoder



# model = CrossEncoder('cross-encoder/stsb-roberta-base')  # or stsb-roberta-large

# # Adelaide Anne Procter ( 30 October 1825 \u2013 2 February 1864 ) was an 

# ex1 = """Q: What was the location of George Calvert's barony after he was created Baron Baltimore in the Irish peerage? """
# ex2 = "Baltimore Manor"

# score =  model.predict([(ex1, ex2)])
# print(ex1,ex2)
# print(score)




     # After you're done with the model
from RACDH.data_generation.target_model import taget_model
from RACDH.data_generation.instruct_model import instruct_model
import torch
del instruct_model
del taget_model
torch.cuda.empty_cache()