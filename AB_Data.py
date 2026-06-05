import csv
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset

from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdmolops
from rdkit.Chem import FragmentCatalog
from rdkit.Chem import MACCSkeys 
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler



class AB_Data:
    def __init__(self,csv_path,pkl_paths):
        self.csv_path = csv_path
        self.heavy_path = pkl_paths[0]
        self.light_path = pkl_paths[1]
        self.antigen_path = pkl_paths[2]
        self.df = pd.read_excel(self.csv_path)

        self.heavy_col = "Antibody Heavy Chain Sequence" 
        self.light_col = "Antibody Light Chain Sequence"
        self.antigen_col = "Antigen Sequence"

        self.str2num = {'<pad>':0 ,'H': 1, 'C': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'P':  9,
         'I': 10,'Na': 11,'B':12,'Se':13,'Si':14,'<unk>':15,'<mask>':16,'<global>':17}
        self.num2str =  {i:j for j,i in self.str2num.items()}

    def calc_aac(self, seq):
        if pd.isna(seq) or seq == "":
            return torch.zeros(20)
        aa_order = 'ACDEFGHIKLMNPQRSTVWY'
        seq = seq.upper()
        length = len(seq)
        if length == 0: return torch.zeros(20)
        counts = [seq.count(aa) for aa in aa_order]
        aac = [c / length for c in counts]
        return torch.tensor(aac, dtype=torch.float32)


    def calc_maccs(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return torch.zeros(167)
            print("MOL IS NONE")
        maccs = MACCSkeys.GenMACCSKeys(mol)
        return torch.tensor(list(map(int, maccs.ToBitString())), dtype=torch.float32)

    def cover_dict(self,path):
        file_path = path
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            tensor_dict = {key: torch.tensor(value) for key, value in data.items()}
            return tensor_dict
            # new_data = {i : value for i,(key,value) in enumerate(tensor_dict.items())}
            # return new_data


    def DAR_feature(self,file_path, column_name):
        df = pd.read_excel(file_path)
        column_data = df[column_name].values.reshape(-1, 1)
        mean_value = 3.86845977
        variance_value = 1.569108443
        std_deviation = variance_value**0.5
        column_data_standardized = (column_data - mean_value) / std_deviation
        normalized_data = (column_data_standardized - 0.8) / (12 - 0.8)
        data_dict = {adc_id: torch.tensor(value, dtype=torch.float32) for adc_id, value in zip(df["ADC ID"], normalized_data.flatten())}
        return data_dict


    def get_dataloaders_from_csv(self, train_csv='data/train.csv', val_csv='data/val.csv', test_csv='data/test.csv', batch_size=40):
            """
            Load train/val/test splits from CSV files instead of random splitting.
            CSV files should contain 'ADC ID' column.
            """
            self.batch_size = batch_size
     
            # Load CSVs
            print("\nLoading pre-split data from CSV files...")
            train_df = pd.read_csv(train_csv)
            val_df = pd.read_csv(val_csv)
            test_df = pd.read_csv(test_csv)
     
            # Extract IDs
            train_ids = train_df['ADC ID'].tolist()
            val_ids = val_df['ADC ID'].tolist()
            test_ids = test_df['ADC ID'].tolist()
     
            print(f"CSV splits - Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
     
            # Load embeddings
            heavy_dict = self.cover_dict(self.heavy_path)
            light_dict = self.cover_dict(self.light_path)
            antigen_dict = self.cover_dict(self.antigen_path)
            dar_val = self.DAR_feature(self.csv_path, "DAR_val")
     
            # Preprocessing for linkers and payloads
            payload_dict = {k: v for k, v in zip(self.df["ADC ID"], self.df["Payload Isosmiles"])}
            linker_dict = {k: v for k, v in zip(self.df["ADC ID"], self.df["Linker Isosmiles"])}
            
            aac_heavy_dict = {k: self.calc_aac(v) for k, v in zip(self.df["ADC ID"], self.df[self.heavy_col])}
            aac_light_dict = {k: self.calc_aac(v) for k, v in zip(self.df["ADC ID"], self.df[self.light_col])}
            aac_antigen_dict = {k: self.calc_aac(v) for k, v in zip(self.df["ADC ID"], self.df[self.antigen_col])}
            maccs_payload_dict = {k: self.calc_maccs(v) for k, v in zip(self.df["ADC ID"], self.df["Payload Isosmiles"])}
            maccs_linker_dict = {k: self.calc_maccs(v) for k, v in zip(self.df["ADC ID"], self.df["Linker Isosmiles"])}
     
            light_dict = {k: v for k, v in light_dict.items() if k in heavy_dict.keys()}
            antigen_dict = {k: v for k, v in antigen_dict.items() if k in heavy_dict.keys()}
            dar_dict = {k: v for k, v in dar_val.items() if k in heavy_dict.keys()}
            payload_dict = {k: v for k, v in payload_dict.items() if k in heavy_dict.keys()}
            linker_dict = {k: v for k, v in linker_dict.items() if k in heavy_dict.keys()}
     
            # Smiles encoding
            main_linker_dict = {k: self.numerical_smiles(v)[0] for k, v in linker_dict.items()}
            main_payload_dict = {k: self.numerical_smiles(v)[0] for k, v in payload_dict.items()}
     
            # Build labels dict
            labels = {}
            for index, row in self.df.iterrows():
                if row["ADC ID"] in heavy_dict.keys():
                    labels[row["ADC ID"]] = row["label（1000nm）"]
     
            def create_dataset_from_ids(id_list, split_name="Split"):
                t1, t2, t3, t4, x1, x2, labels_list = [], [], [], [], [], [], []
                aac1_list, aac2_list, aac3_list = [], [], []
                x1_maccs_list, x2_maccs_list = [], []
                 
                dropped_missing_heavy = []
     
                for adc_id in id_list:
                    if adc_id not in heavy_dict.keys():
                        dropped_missing_heavy.append(adc_id)
                        continue
     
                    t1.append(heavy_dict[adc_id].squeeze())
                    t2.append(light_dict[adc_id])
                    t3.append(antigen_dict[adc_id])
                    t4.append(dar_dict[adc_id])
                    x1.append(main_payload_dict[adc_id])
                    x2.append(main_linker_dict[adc_id])
                    labels_list.append(labels[adc_id])
                    aac1_list.append(aac_heavy_dict.get(adc_id, torch.zeros(20)))
                    aac2_list.append(aac_light_dict.get(adc_id, torch.zeros(20)))
                    aac3_list.append(aac_antigen_dict.get(adc_id, torch.zeros(20)))
                    x1_maccs_list.append(maccs_payload_dict.get(adc_id, torch.zeros(167)))
                    x2_maccs_list.append(maccs_linker_dict.get(adc_id, torch.zeros(167)))
     
                # --- DEBUG: Print Report ---
                if dropped_missing_heavy:
                    print(f"\n[DEBUG] Dropped samples in {split_name}:")
                    print(f"  - {len(dropped_missing_heavy)} samples dropped because missing in 'heavy_dict' (pkl): {dropped_missing_heavy}")
                # ---------------------------
     
                x1 = [torch.tensor(x) for x in x1]
                x2 = [torch.tensor(x) for x in x2]
                t4 = torch.tensor(t4)
     
                x1 = [F.pad(x, pad=(0, 195 - x.shape[0])) for x in x1]
                x2 = [F.pad(x, pad=(0, 184 - x.shape[0])) for x in x2]
                x1 = torch.stack(x1)
                x2 = torch.stack(x2)
     
                t1 = torch.tensor(np.array([x.cpu() for x in t1]))
                t2 = torch.tensor(np.array([x.cpu() for x in t2]))
                t3 = torch.tensor(np.array([x.cpu() for x in t3]))
                t4 = torch.tensor(np.array([x.cpu() for x in t4]))
                
                aac1 = torch.stack(aac1_list)
                aac2 = torch.stack(aac2_list)
                aac3 = torch.stack(aac3_list)
                x1_maccs = torch.stack(x1_maccs_list)
                x2_maccs = torch.stack(x2_maccs_list)
                
                labels_list = torch.tensor(labels_list)
     
                return TensorDataset(x1, x1_maccs, x2, x2_maccs, t1, t2, t3, aac1, aac2, aac3, t4, labels_list)
     
            # Create datasets
            train_dataset = create_dataset_from_ids(train_ids, "Train")
            val_dataset = create_dataset_from_ids(val_ids, "Val")
            test_dataset = create_dataset_from_ids(test_ids, "Test")
     
            print(f"Datasets created - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
     
            # Create dataloaders
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
     
            self.train_dataset = train_dataset
            self.valid_dataset = val_dataset
            self.test_dataset = test_dataset
            self.train_loader = train_loader
            self.valid_loader = val_loader
            self.test_loader = test_loader
     
            return train_loader, val_loader, test_loader
     
     
    def get_dataloaders(self,seed,train_ratio=0.8,valid_ratio=0.1,test_ratio= 0.1,batch_size = 40,use_ddp = False):
        self.train_ratio = train_ratio
        self.seed = seed
        self.batch_size = batch_size
        
        heavy_dict = self.cover_dict(self.heavy_path)
        light_dict = self.cover_dict(self.light_path)
        antigen_dict = self.cover_dict(self.antigen_path)
        dar_val = self.DAR_feature(self.csv_path,"DAR_val")

        payload_dict = {k:v for k,v in zip(self.df["ADC ID"],self.df["Payload Isosmiles"])}
        linker_dict = {k:v for k,v in zip(self.df["ADC ID"],self.df["Linker Isosmiles"])}
        
        aac_heavy_dict = {k: self.calc_aac(v) for k, v in zip(self.df["ADC ID"], self.df[self.heavy_col])}
        aac_light_dict = {k: self.calc_aac(v) for k, v in zip(self.df["ADC ID"], self.df[self.light_col])}
        aac_antigen_dict = {k: self.calc_aac(v) for k, v in zip(self.df["ADC ID"], self.df[self.antigen_col])}
        maccs_payload_dict = {k: self.calc_maccs(v) for k, v in zip(self.df["ADC ID"], self.df["Payload Isosmiles"])}
        maccs_linker_dict = {k: self.calc_maccs(v) for k, v in zip(self.df["ADC ID"], self.df["Linker Isosmiles"])}


        light_dict = {k : v for k,v in light_dict.items() if k in heavy_dict.keys()}
        antigen_dict = {k : v for k,v in antigen_dict.items() if k in heavy_dict.keys()}
        dar_dict = {k: v for k,v in dar_val.items() if k in heavy_dict.keys()}
        
        payload_dict = {k: v for k,v in payload_dict.items() if k in heavy_dict.keys()}
        linker_dict = {k: v for k,v in linker_dict.items() if k in heavy_dict.keys()}


        #Smiles encoding
        main_linker_dict = {k: self.numerical_smiles(v)[0] for k,v in linker_dict.items()}
        main_payload_dict = {k: self.numerical_smiles(v)[0] for k,v in payload_dict.items()}


        main_adjoint_linker_dict = {v : self.numerical_smiles(v)[1] for k,v in linker_dict.items()}
        main_adjoint_payload_dict = {v: self.numerical_smiles(v)[1] for k,v in payload_dict.items()}
        main_adjoint_dict = main_adjoint_linker_dict | main_adjoint_payload_dict
        
        t1,t2,t3,t4,x1,x2 = [],[],[],[],[],[]
        aac1_list, aac2_list, aac3_list = [], [], []
        x1_maccs_list, x2_maccs_list = [], []
        
        labels = {}
        for index,row in self.df.iterrows():
             if row["ADC ID"] in heavy_dict.keys():
                 labels[row["ADC ID"]] = row["label（1000nm）"]
        labels_list = []
        for ele in self.df["ADC ID"]:
            if ele in set(heavy_dict.keys()):
                t1.append(heavy_dict[ele].squeeze())
                t2.append(light_dict[ele])
                t3.append(antigen_dict[ele])
                t4.append(dar_dict[ele])
                x1.append(main_payload_dict[ele])
                x2.append(main_linker_dict[ele])
                labels_list.append(labels[ele])
                
                aac1_list.append(aac_heavy_dict.get(ele, torch.zeros(20)))
                aac2_list.append(aac_light_dict.get(ele, torch.zeros(20)))
                aac3_list.append(aac_antigen_dict.get(ele, torch.zeros(20)))
                x1_maccs_list.append(maccs_payload_dict.get(ele, torch.zeros(167)))
                x2_maccs_list.append(maccs_linker_dict.get(ele, torch.zeros(167)))
        
        
        
        x1 = [torch.tensor(x) for x in x1]
        x2 = [torch.tensor(x) for x in x2]

        
        for i, x in enumerate(x1[:10]):
            print(i, type(x), x.shape)

        
        for i, x in enumerate(x2[:10]):
            print(i, type(x), x.shape)

        t4 = torch.tensor(t4)

        x1 = [F.pad(x, pad=(0, 195 - x.shape[0])) for x in x1]
        x2 = [F.pad(x, pad=(0, 184 - x.shape[0])) for x in x2]



        x1 = torch.stack(x1)
        x2 = torch.stack(x2)        

        
        for i, x in enumerate(t1[:20]):
            print(i, type(x), x.shape)

        print(set([tuple(x.shape) for x in t1]))
        
        t1 = torch.stack([x.cpu().float() for x in t1])
        t2 = torch.stack([
            x.cpu().float() if isinstance(x, torch.Tensor)
            else torch.tensor(x).float()
            for x in t2
        ])

        t3 = torch.stack([
            x.cpu().float() if isinstance(x, torch.Tensor)
            else torch.tensor(x).float()
            for x in t3
        ])

        t4 = torch.stack([
            x.cpu().float() if isinstance(x, torch.Tensor)
            else torch.tensor(x).float()
            for x in t4
        ])
        
        aac1 = torch.stack(aac1_list)
        aac2 = torch.stack(aac2_list)
        aac3 = torch.stack(aac3_list)
        x1_maccs = torch.stack(x1_maccs_list)
        x2_maccs = torch.stack(x2_maccs_list)

        labels_list = torch.tensor(labels_list)

        dataset = TensorDataset(x1, x1_maccs, x2, x2_maccs, t1, t2, t3, aac1, aac2, aac3, t4, labels_list)
        train_size = int(train_ratio * len(dataset))
        valid_size = int(valid_ratio * len(dataset))
        test_size = len(dataset) - train_size - valid_size 
    
        generator = torch.Generator().manual_seed(self.seed)
                
        print(train_ratio,valid_ratio,test_ratio)        
        train_dataset,valid_dataset, test_dataset = random_split(dataset, [train_size,valid_size,test_size],generator = generator)

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        train_dataloader = DataLoader(train_dataset,batch_size = self.batch_size,shuffle = True)
        valid_dataloader = DataLoader(valid_dataset, batch_size = self.batch_size, shuffle = False)
        test_dataloader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle = False)

        print(len(train_dataset))
        print(len(valid_dataset))
        print(len(test_dataset))
            
        self.train_loader = train_dataloader
        self.test_loader = test_dataloader
        self.valid_loader = valid_dataloader

        return train_dataloader, valid_dataloader , test_dataloader
            
    def smiles2adjoin(self,smiles,explicit_hydrogens=True,canonical_atom_order=False): 

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
          print('error')
          return None,None
    
        if explicit_hydrogens:
            mol = Chem.AddHs(mol)
        else:
            mol = Chem.RemoveHs(mol)
    
        if canonical_atom_order:
            new_order = rdmolfiles.CanonicalRankAtoms(mol)
            mol = rdmolops.RenumberAtoms(mol, new_order)
        num_atoms = mol.GetNumAtoms()
    
        atoms_list = []
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            atoms_list.append(atom.GetSymbol())
    
    
        adjoin_matrix = np.eye(num_atoms)
        num_bonds = mol.GetNumBonds()
    
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            adjoin_matrix[u,v] = 1.0
            adjoin_matrix[v,u] = 1.0
    
        return atoms_list,adjoin_matrix
    
    def numerical_smiles(self,smiles):
        smiles_origin = smiles
        atoms_list, adjoin_matrix = self.smiles2adjoin(smiles,explicit_hydrogens = True)
        atoms_list = ["<global>"] + atoms_list
        nums_list = [self.str2num.get(atom, self.str2num['<unk>']) for atom in atoms_list]
    
        temp = np.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)
    
        x = np.array(nums_list, dtype=np.int64)
        return x, adjoin_matrix, smiles_origin, atoms_list
