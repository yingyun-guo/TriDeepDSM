# -*- coding: utf-8 -*-
import sys
import torch
import numpy as np
import json
from Bio import SeqIO
from pathlib import Path

sys.path.append('/data/gyy/Project/Feature-Tool/hyena-dna-main/')
try:
    from standalone_hyenadna import HyenaDNAModel
    from huggingface import CharacterTokenizer
except ImportError as e:
    print(f"导入错误: {e}. 请确保 standalone_hyenadna.py 和 huggingface.py 存在")
    sys.exit(1)


def load_model_and_tokenizer(model_path, model_name='hyenadna-small-32k-seqlen', max_length=32768):
    try:
        config_path = Path(model_path) / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_config = config.get('model', config)
        model = HyenaDNAModel(**model_config)
        checkpoint_path = Path(model_path) / 'weights.ckpt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')  # 强制 CPU
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        tokenizer = CharacterTokenizer(
            characters=['A', 'C', 'G', 'T', 'N'],
            model_max_length=max_length,
        )
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)


def extract_sequences(fasta_path):
    try:
        sequences = [str(record.seq).upper() for record in SeqIO.parse(fasta_path, 'fasta')]
        print(f"从 {fasta_path} 加载了 {len(sequences)} 条序列")
        return sequences
    except Exception as e:
        print(f"FASTA 文件解析失败: {e}")
        sys.exit(1)


def get_embeddings(model, tokenizer, sequences, batch_size=8, use_padding=True, add_eos=False):
    embeddings = []
    device = 'cpu'  # 强制 CPU
    model.to(device)
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            try:
                inputs = tokenizer(
                    batch_seqs,
                    return_tensors='pt',
                    padding=use_padding,
                    add_special_tokens=add_eos,
                    max_length=tokenizer.model_max_length,
                    truncation=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(inputs['input_ids'])
                emb = outputs.mean(dim=1).cpu().numpy()
                embeddings.append(emb)
                print(f"处理批次 {i // batch_size + 1}/{total_batches}, 嵌入形状: {emb.shape}")
            except Exception as e:
                print(f"批处理 {i // batch_size + 1} 失败: {e}")
                continue
    if not embeddings:
        print("未生成任何嵌入")
        sys.exit(1)
    return np.vstack(embeddings)


if __name__ == '__main__':
    model_path = '/data/gyy/Project/Feature-Tool/hyena-dna-main/hyenadna-small-32k-seqlen/'
    fasta_train = '/data/gyy/Project/EPEL-main/data/COSMIC_training.fasta'
    fasta_test = '/data/gyy/Project/EPEL-main/data/COSMIC_testing.fasta'
    output_dir = Path('/data/gyy/Project/EPEL-main/features/')
    output_dir.mkdir(exist_ok=True)
    model_name = 'hyenadna-small-32k-seqlen'
    max_length = 32768

    model, tokenizer = load_model_and_tokenizer(model_path, model_name, max_length)

    train_seqs = extract_sequences(fasta_train)
    train_emb = get_embeddings(model, tokenizer, train_seqs)
    train_path = output_dir / 'COSMIC_training_embeddings.npy'
    np.save(train_path, train_emb)
    print(f"训练集特征形状: {train_emb.shape}, 已保存到 {train_path}")

    test_seqs = extract_sequences(fasta_test)
    test_emb = get_embeddings(model, tokenizer, test_seqs)
    test_path = output_dir / 'COSMIC_testing_embeddings.npy'
    np.save(test_path, test_emb)
    print(f"测试集特征形状: {test_emb.shape}, 已保存到 {test_path}")



