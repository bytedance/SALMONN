import torch
from torch.utils.data import Sampler
import random

class WeightedRoundRobinBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.generator = None  # 将在每个 epoch 初始化

        # 分组（只做一次）
        self.indices = {
            "av": [i for i in range(len(dataset)) if dataset.type_list[i] == "av"],
            "v": [i for i in range(len(dataset)) if dataset.type_list[i] == "v"],
            "a": [i for i in range(len(dataset)) if dataset.type_list[i] == "a"],
            "t": [i for i in range(len(dataset)) if dataset.type_list[i] == "t"],
        }

        print(f"av: {len(self.indices['av'])}")
        print(f"v: {len(self.indices['v'])}")
        print(f"a: {len(self.indices['a'])}")
        print(f"t: {len(self.indices['t'])}")

        # 计算权重（比例 ~ 数据量大小）
        self.weights = {k: len(v) for k, v in self.indices.items()}
        total = sum(self.weights.values())
        self.probs_template = {k: self.weights[k] / total for k in self.weights}  # 原始概率模板

        # 初始化状态
        self.cursors = None
        self.probs = None

        self.epoch = 0

        self._shuffle_data()

    def _shuffle_data(self):
        """每个 epoch 开始时调用，打乱各组数据"""
        self.generator = torch.Generator().manual_seed(self.seed + self.epoch if hasattr(self, 'epoch') else self.seed)
        for k in self.indices:
            idxs = torch.tensor(self.indices[k])
            shuffled = idxs[torch.randperm(len(idxs), generator=self.generator)].tolist()
            self.indices[k] = shuffled
        self.cursors = {k: 0 for k in self.indices}
        self.probs = self.probs_template.copy()  # 重置为原始概率

        exhausted = False
        self.out_data = []
        while not exhausted:
            # 按照概率决定哪个 modality 出现
            modalities = list(self.probs.keys())
            probs = torch.tensor([self.probs[m] for m in modalities], dtype=torch.float32)
            idx = torch.multinomial(probs, num_samples=1, generator=self.generator).item()
            m = modalities[idx]

            start = self.cursors[m]
            end = start + self.batch_size
            if end <= len(self.indices[m]):
                batch = self.indices[m][start:end]
                self.cursors[m] = end
                self.out_data.append(batch)
            else:
                # 如果该 modality 数据用完，就把它的概率置 0
                self.probs[m] = 0
                total = sum(self.probs.values())
                if total == 0:
                    exhausted = True
                else:
                    # 重新归一化
                    self.probs = {k: v / total for k, v in self.probs.items()}

    def __iter__(self):
        for batch in self.out_data:
            for b in batch:
                yield b
        

    def __len__(self):
        # 返回理论最大 batch 数（向下取整）
        return sum(len(batch) for batch in self.out_data)

    def set_epoch(self, epoch):
        """供 DistributedSampler 或手动调用，设置 epoch 以改变 shuffle 种子"""
        self.epoch = epoch
        self._shuffle_data()