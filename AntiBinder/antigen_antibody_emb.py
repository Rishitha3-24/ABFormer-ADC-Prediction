import esm
import os
import lmdb
import pickle
import torch
import torch.nn as nn
import torch.nn. functional as F
import pandas as pd
import sys
# sys.path.append("/AntiBinder")
from AntiBinder.cfg_ab import AminoAcid_Vocab
from AntiBinder.cfg_ab import configuration
import pdb
from math import ceil
from igfold import IgFoldRunner
import hashlib
from torch.utils.data import Dataset

class antibody_antigen_dataset(Dataset):
    def __init__(self,
                antigen_config: configuration, 
                antibody_config: configuration, 
                data_path=None,
                train = True,
                test = False, 
                rate1 = 0.8,
                data = None) -> None:
        super().__init__()
        self.antigen_config = antigen_config
        self.antibody_config = antibody_config
        data_path = "main.csv"
        if isinstance(data,pd.DataFrame):
            df = data
            df.dropna()
        else:
            df = pd.read_csv(data_path)
                        
            ## samples of data, attention to our file type
            # df = df.dropna()

        print(df.shape)
        df = df.dropna(subset=['H-FR1','H-CDR1','H-FR2','H-CDR2','H-FR3','H-CDR3','H-FR4'])
            
        self.antigen_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = alphabet.get_batch_converter()
        self.data = df
        # if train==True and test==False: # train
        #     print("This part of dataset is for train.")
        #     self.data = df.iloc[:int(df.shape[0]*(rate1))]
            
        #     print("len train data", len(self.data)) 

        # elif train==False and test == True: # val
        #     print("This part of dataset is for test.")
        #     self.data = df.iloc[int(df.shape[0]*(rate1)):]
        #     print("len test data",len(self.data))


        self.structure_embedding = None
        self.igfold = IgFoldRunner()


    def universal_padding(self, sequence, max_length):
        if len(sequence) > max_length:
            return sequence[:max_length]
        else:
            return torch.cat([sequence,torch.zeros(max_length-len(sequence))]).long()
    

    def func_padding_for_esm(self, sequence, max_length):
        if len(sequence) > max_length:
            return sequence[:max_length]
        else:
            return sequence+'<pad>'*(max_length-len(sequence)-2)
        

    def region_indexing(self,index):
        data = self.data.iloc[index]
        HF1 = [1 for _ in range(len(data['H-FR1']))]
        HCDR1 = [3 for _ in range(len (data['H-CDR1']))]
        HF2 = [1 for _ in range(len(data['H-FR2']))]
        HCDR2 = [4 for _ in range(len(data['H-CDR2']))]
        HF3 = [1 for _ in range(len(data['H-FR3']))]
        HCDR3 = [5 for _ in range(len(data['H-CDR3']))]
        HF4 = [1 for _ in range(len(data['H-FR4']))]
        vh = torch.tensor(list(HF1+HCDR1+HF2+HCDR2+HF3+HCDR3+HF4))
        vh = self.universal_padding(vh,self.antibody_config.max_position_embeddings)

        return vh
    

    def __getitem__(self, index):
        # if index == 0: 
        #     if os.path.exists("datasets/fold_emb/fold_emb_for_train"):
        #         shutil.rmtree("datasets/fold_emb/fold_emb_for_train")
        data = self.data.iloc[index]
        # label = torch.tensor(data['ANT_Binding'])


        name = str(self.data.iloc[index]['Antigen Sequence'])
        unique_name = hashlib.md5(name.encode()).hexdigest()
        
        file_path = 'Embeddings/antigen_esm/' + unique_name + ".pt"
        if not os.path.exists(file_path):
            antigen = self.func_padding_for_esm(self.data['Antigen Sequence'].iloc[index], self.antigen_config.max_position_embeddings)
            antigen = [('antigen', antigen)]
            batch_labels, batch_strs, antigen = self.batch_converter(antigen)
            with torch.no_grad():
                self.antigen_model = self.antigen_model.eval()
                antigen = self.antigen_model(antigen.squeeze(1), repr_layers=[33], return_contacts=True)
                antigen = antigen['representations'][33].squeeze(0)
            torch.save(antigen,file_path)
            print(f"{index} done.")
        
        antigen_structure = torch.load(file_path)
        # print("antigen_structure："，antigen_structure)
        # print("antigen_structure shape: ", antigen_structure.shape)
        antibody_name = data['vh']
        antigen_name = data['Antigen Sequence']
        emb_seq = data['H-FR1'] + data['H-CDR1'] + data['H-FR2'] + data['H-CDR2'] + data['H-FR3'] + data['H-CDR3']+data['H-FR4']
        #if not emb_seq in self.structure_embedding.keys():

        self.env = lmdb.open(f'Embeddings',map_size=1024*1024*1024*50,lock=True)
        self.structure_embedding = self.env.begin(write=True)
        if self.structure_embedding.get(emb_seq.encode()) == None:
            sequences = {"H": emb_seq}
            emb = self.igfold.embed(sequences=sequences)
            structure = emb.structure_embs.detach().cpu()
            self.structure_embedding.put(key=emb_seq.encode(), value=pickle.dumps(structure)) 
            self.structure_embedding.commit()
        else:
            structure = pickle.loads(self.structure_embedding.get(emb_seq.encode()))
        self.env.close()


        structure_m1 = len(data['H-FR1'] + data['H-CDR1'] + data['H-FR2'] + data['H-CDR2'] + data['H-FR3'])
        structure_m2 = len(data['H-FR1'] + data['H-CDR1'] + data['H-FR2'] + data['H-CDR2'] + data['H-FR3']+ data['H-CDR3'])
        structure_m3 = len(data['H-FR1'] + data['H-CDR1'] + data['H-FR2'] + data['H-CDR2'] + data['H-FR3']+ data['H-CDR3'] + data['H-FR4'])
        # print("zero_for_padding", zero_for_padding)
        # print("zero_for_padding.shape"，zero_for_padding.shape)
        part1_indices = torch.arange(structure_m1)
        part2_indices = torch.arange(structure_m1, structure_m2)
        part3_indices = torch.arange(structure_m2, structure_m3)

        part1 = structure.index_select(1, part1_indices)
        part2_full = structure.index_select(1, part2_indices)
        part2_mean = part2_full.mean(dim=1)
        part2_reshaped = part2_mean.view(part2_mean.size(0), 1, part2_mean.size(1))
        part3 = structure.index_select(1, part3_indices)

        structure = torch.cat((
            part1.detach().clone(),
            part2_reshaped.detach().clone(),
            part3.detach().clone()
        ), dim=1)
        # print("stru.shape", structure.shape)
        zero_for_padding = torch.zeros(1,self.antibody_config.max_position_embeddings-structure.shape[1], structure.shape[-1]).float()
        structure = torch.cat((structure, zero_for_padding) ,dim=1).squeeze(0)


        antibody = data['vh']
        # print("antibody："，antibody)
        antibody = torch.tensor([AminoAcid_Vocab[aa] for aa in antibody])
        # print("antibody："，antibody)
        antibody = self.universal_padding(sequence=antibody, max_length=self.antibody_config.max_position_embeddings)
        # print("antibody: ", antibody)
        # print("antibody_shape:", antibody. shape)

        at_type = self.region_indexing(index)
        # print(type)
        # print(type.shape)
        antibody_structure = structure

        antigen = data['Antigen Sequence']
        antigen = torch.tensor([AminoAcid_Vocab[aa] for aa in antigen])
        antigen = self.universal_padding(sequence=antigen, max_length=self.antigen_config.max_position_embeddings)

        antigen_structure = antigen_structure[:1024, :]
        # print(antigen_structure.shape)
        return [antibody,at_type,antibody_structure],[antigen,antigen_structure]

    def __len__(self):
        return self.data.shape[0]

