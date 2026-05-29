import os
import sys
import json
import warnings
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import esm
from igfold import IgFoldRunner
from abnumber import Chain
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
from rdkit.Chem import MACCSkeys

from transformers import BertConfig, BertTokenizer
from transformers.tokenization_utils import Trie
from transformers.models.bert.tokenization_bert import *

from ADCNet import PredictModel
from AntiBinder.antibinder_model import *
from AntiBinder.antigen_antibody_emb import *
from AntiBinder.cfg_ab import *

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.serialization.add_safe_globals([
    BertConfig, BertTokenizer, Trie, BasicTokenizer, WordpieceTokenizer,
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading ESM2...")
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
esm_model = esm_model.to(device).eval()

print("Loading IgFold...")
igfold = IgFoldRunner()

def calc_aac(seq):
    aa_order = 'ACDEFGHIKLMNPQRSTVWY'
    if seq is None or len(seq) == 0:
        return torch.zeros(20)
    seq = seq.upper()
    length = len(seq)
    counts = [seq.count(aa) for aa in aa_order]
    return torch.tensor([c / length for c in counts], dtype=torch.float32)

def calc_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(167)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    return torch.tensor(list(map(int, maccs.ToBitString())), dtype=torch.float32)

def split_heavy_chain_regions(sequence, scheme='chothia'):
    REGION_MAP = {
        'H-FR1': 'fr1_seq', 'H-CDR1': 'cdr1_seq', 'H-FR2': 'fr2_seq',
        'H-CDR2': 'cdr2_seq', 'H-FR3': 'fr3_seq', 'H-CDR3': 'cdr3_seq',
        'H-FR4': 'fr4_seq'
    }
    out = {r: '' for r in REGION_MAP}
    try:
        ch = Chain(sequence, scheme=scheme)
        if ch.chain_type == 'H':
            for r, a in REGION_MAP.items():
                out[r] = getattr(ch, a) or ''
    except:
        pass
    return out

def universal_padding(sequence, max_len):
    seq_len = len(sequence)
    if seq_len > max_len:
        return sequence[:max_len]
    pad = torch.zeros(max_len - seq_len, dtype=torch.long)
    return torch.cat([sequence, pad])

def func_padding_for_esm(sequence, max_length):
    if len(sequence) >= max_length:
        return sequence[:max_length]
    return sequence + "<pad>" * (max_length - len(sequence))

def region_indexing(data):
    region_list = ['H-FR1','H-CDR1','H-FR2','H-CDR2','H-FR3','H-CDR3','H-FR4']
    mapping = {'FR': 1, 'CDR1': 3, 'CDR2': 4, 'CDR3': 5}
    idx = []
    for r in region_list:
        rt = r.split("-")[1]
        if "CDR" in rt:
            idx.extend([mapping[rt]] * len(data[r]))
        else:
            idx.extend([mapping['FR']] * len(data[r]))
    return universal_padding(torch.tensor(idx), 149)

str2num = {
    '<pad>':0, 'H':1,'C':2,'N':3,'O':4,'S':5,'F':6,'Cl':7,'Br':8,'P':9,
    'I':10,'Na':11,'B':12,'Se':13,'Si':14,'<unk>':15,'<mask>':16,'<global>':17
}

def smiles2adjoin(smiles, explicit_h=True, canonical=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], np.zeros((1,1))
    mol = Chem.AddHs(mol) if explicit_h else Chem.RemoveHs(mol)
    if canonical:
        order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, order)
    n = mol.GetNumAtoms()
    atoms = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n)]
    adj = np.eye(n)
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[u,v] = adj[v,u] = 1
    return atoms, adj

def numerical_smiles(smiles):
    atoms, adj = smiles2adjoin(smiles, explicit_h=True)
    if len(atoms) == 0:
        return np.array([0])
    atoms = ["<global>"] + atoms
    nums = [str2num.get(a, str2num['<unk>']) for a in atoms]
    n = len(nums)
    temp = np.ones((n,n))
    temp[1:,1:] = adj
    adj_final = (1 - temp) * (-1e9)
    return np.array(nums, dtype=np.int64)

def get_esm2_embeddings(data):
    labels, strs, toks = batch_converter(data)
    toks = toks.to(device)
    with torch.no_grad():
        out = esm_model(toks, repr_layers=[33], return_contacts=False)
        rep = out["representations"][33]
    embs = []
    for i, (_, seq) in enumerate(data):
        emb = rep[i, 1:len(seq)+1].mean(dim=0, keepdim=True)
        embs.append(emb)
    return embs

def process_antibody_antigen(sequences, new_dict, antigen_structure):
    emb = igfold.embed(sequences=sequences)
    struct = emb.structure_embs.detach().cpu()

    heavy = ''.join(new_dict[r] for r in
                    ['H-FR1','H-CDR1','H-FR2','H-CDR2','H-FR3','H-CDR3','H-FR4'])
    pad = 149 - len(heavy)
    if pad < 0:
        struct = struct[:, :149, :]
    else:
        z = torch.zeros(1, pad, struct.size(-1))
        struct = torch.cat([struct, z], dim=1)

    ab = torch.tensor([AminoAcid_Vocab.get(a,0) for a in new_dict['t1']])
    ab = universal_padding(ab,149)
    at = region_indexing(new_dict)

    ag_seq = new_dict["t3"].strip()
    ag = torch.tensor([AminoAcid_Vocab.get(a,0) for a in ag_seq])
    ag = universal_padding(ag,1024)

    L, D = antigen_structure.shape
    if L > 1024:
        antigen_structure = antigen_structure[:1024,:]
    else:
        antigen_structure = F.pad(antigen_structure,(0,0,0,1024-L))

    antibody = [ab.unsqueeze(0), at.unsqueeze(0), struct]
    antigen = [ag.unsqueeze(0), antigen_structure.unsqueeze(0)]
    return antibody, antigen

def DAR_feature(v):
    mean = 3.86845977
    var  = 1.569108443
    std  = var ** 0.5
    x = torch.tensor(v, dtype=torch.float32)
    return ((x - mean)/std - 0.8) / (12 - 0.8)

# ----------------------------------------------------------
# Single–sample preprocessing (JSON)
# ----------------------------------------------------------
def preprocess_single_sample(sample, AntiBinder_model):
    H = sample["Heavy_Chain_Sequence"]
    L = sample["Light_Chain_Sequence"]
    A = sample["Antigen_Sequence"]
    P = sample["Payload_SMILES"]
    K = sample["Linker_SMILES"]
    DAR = DAR_feature(sample["DAR"])

    new = {'t1': H, 't2': L, 't3': A, 't4': DAR}
    regions = split_heavy_chain_regions(H)
    new.update(regions)

    antigen_embed, light_embed = get_esm2_embeddings(
        [('antigen',A),('light_chain',L)]
    )

    padded = func_padding_for_esm(A,1024)
    toks = batch_converter([('antigen',padded)])[2].to(device)
    antigen_struct = esm_model(toks, repr_layers=[33])['representations'][33].squeeze(0)

    heavy_seq = ''.join(new[r] for r in
                        ['H-FR1','H-CDR1','H-FR2','H-CDR2','H-FR3','H-CDR3','H-FR4'])
    if len(heavy_seq) < 5:
        heavy_seq = H

    ab_set, ag_set = process_antibody_antigen({'H':heavy_seq}, new, antigen_struct)
    ab_set = [t.to(device) for t in ab_set]
    ag_set = [t.to(device) for t in ag_set]

    with torch.no_grad():
        heavy_emb = AntiBinder_model(ab_set, ag_set)

    payload_num = numerical_smiles(P)
    linker_num  = numerical_smiles(K)

    x1 = F.pad(torch.from_numpy(payload_num), (0,195-len(payload_num))).unsqueeze(0).to(device).int()
    x2 = F.pad(torch.from_numpy(linker_num),  (0,184-len(linker_num))).unsqueeze(0).to(device).int()

    x1m = calc_maccs(P).unsqueeze(0).to(device)
    x2m = calc_maccs(K).unsqueeze(0).to(device)

    aac1 = calc_aac(H).unsqueeze(0).to(device)
    aac2 = calc_aac(L).unsqueeze(0).to(device)
    aac3 = calc_aac(A).unsqueeze(0).to(device)

    return {
        'x1': x1, 'x1maccs': x1m,
        'x2': x2, 'x2maccs': x2m,
        't1': heavy_emb.float(),
        't2': light_embed.float(),
        't3': antigen_embed.float(),
        't4': torch.tensor([DAR],device=device).float(),
        'aac1': aac1, 'aac2': aac2, 'aac3': aac3
    }

def main(seed, json_path):
    
    ADC_model = PredictModel(
        num_layers=6, d_model=256, dff=512,
        num_heads=8, vocab_size=18
    ).to(device)

    ckpt = f"ckpts/ADC_{seed}_best_model.pth"
    ADC_model.load_state_dict(torch.load(ckpt, map_location=device))
    ADC_model.eval()

    ab_ckpt = "model_weights/AntiBinder_Weights.pth"
    sd = torch.load(ab_ckpt, map_location=device)
    sd = {k.replace("module.",""):v for k,v in sd.items()}
    AntiBinder_model = antibinder().to(device)
    AntiBinder_model.load_state_dict(sd, strict=False)
    AntiBinder_model.eval()

    with open(json_path,"r") as f:
        sample = json.load(f)

    processed = preprocess_single_sample(sample, AntiBinder_model)

    with torch.no_grad():
        out = ADC_model(
            processed['x1'], processed['x1maccs'],
            processed['x2'], processed['x2maccs'],
            processed['t1'], processed['t2'], processed['t3'],
            processed['aac1'], processed['aac2'], processed['aac3'],
            processed['t4'],
            mask1=None, adjoin_matrix1=None,
            mask2=None, adjoin_matrix2=None
        )

    prob = torch.sigmoid(out).item()
    label = 1 if prob > 0.5 else 0

    print(f"\nPredicted Probability: {prob:.4f}")
    print(f"Predicted Label: {label}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", type=str, default="input.json")
    args = parser.parse_args()
    main(args.seed, args.json)
