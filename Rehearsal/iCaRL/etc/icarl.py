
from math import ceil
import numpy as np

import torch
from torch import nn

class ICaRLPlugin(nn.Module):
    def __init__(self, args, model, feature_size, device, buffer_transform=None, fixed_memory=True):
        super().__init__()

        self.args = args
        self.model = model
        self.class_means = None
        self.memory_size = args.memory
        self.exemplars_per_class = None
        self.embedding_size = feature_size
        self.device = device

        self.x_memory = []
        self.y_memory = []
        self.observed_classes = 0

    def after_training_exp(self, train_loader):
        self.model.eval()

        self.compute_class_means(train_loader)
        self.exemplars_per_class = self.memory_size // self.observed_classes
        if len(self.x_memory) > 0: self.reduce_exemplar_set()
        self.construct_exemplar_set(train_loader)

    def compute_class_means(self, train_loader):
        if self.class_means is None:
            n_classes = self.args.num_classes
            self.class_means = torch.zeros((self.embedding_size, n_classes)).to(self.device)

        extracted_features = []
        extracted_targets = []
        
        with torch.no_grad():
            for images, targets in train_loader:
                images = images.to(self.device)

                features = self.model.features(images)
                extracted_features.append(features / torch.norm(features, dim=0))
                extracted_targets.extend(targets)
            
            extracted_features = torch.cat(extracted_features)
            extracted_targets = np.array(extracted_targets)

            for curr_cls in np.unique(extracted_targets):
                # get all indices from current class
                cls_ind = np.where(extracted_targets == curr_cls)[0]
                # get all extracted features for current class
                cls_feats = extracted_features[cls_ind]
                # add the exemplars to the set and normalize
                cls_feats_mean = cls_feats.mean(0) / cls_feats.mean(0).norm()
                self.class_means[:, curr_cls] = cls_feats_mean

    def reduce_exemplar_set(self):
        x_buffer = []
        y_buffer = []

        for idx, curr_cls in enumerate(np.unique(self.y_memory)):
            # get all indices from current class
            cls_ind = len(np.where(self.y_memory == curr_cls)[0])
            start = idx * cls_ind
            end = (idx + 1) * cls_ind

            x_temp = self.x_memory[start:end]
            y_temp = self.y_memory[start:end]

            x_buffer.append(x_temp[:self.exemplars_per_class])
            y_buffer.extend(y_temp[:self.exemplars_per_class])

        self.x_memory = torch.cat(x_buffer)
        self.y_memory = np.array(y_buffer)
                
    def construct_exemplar_set(self, train_loader):
        extracted_features = []
        extracted_targets = []
        x_buffer = []

        with torch.no_grad():
            for images, targets in train_loader:
                images = images.to(self.device)

                features = self.model.features(images)
                extracted_features.append(features / torch.norm(features, dim=0))
                extracted_targets.extend(targets)
                x_buffer.append(images)
            
            extracted_features = torch.cat(extracted_features)
            extracted_targets = np.array(extracted_targets)
            x_buffer = torch.cat(x_buffer)

            last_idx = len(np.unique(extracted_targets))

            for idx, curr_cls in enumerate(np.unique(extracted_targets)):
                cls_ind = np.where(extracted_targets == curr_cls)[0]

                pred_inter = (extracted_features[cls_ind].mT / torch.norm(extracted_features[cls_ind].T, dim=0)).mT
                mean_temp = self.class_means[:, curr_cls][:, None].permute(1, 0)
                sqd = torch.cdist(mean_temp, pred_inter).squeeze()

                if (idx+1) == last_idx and (idx+1)*len(cls_ind) != self.memory_size:
                    end = self.memory_size - self.x_memory.shape[0]
                    argsort = torch.argsort(sqd)[:end]
                else:
                    argsort = torch.argsort(sqd)[:self.exemplars_per_class]
                sort_ind = cls_ind[argsort.cpu()]

                x_temp = torch.tensor(self.x_memory, device=self.device).clone().detach()
                self.x_memory = torch.cat([x_temp, x_buffer[sort_ind]])
                self.y_memory = np.concatenate([self.y_memory, extracted_targets[sort_ind]])
            
            self.y_memory = np.array(self.y_memory)

    def forward(self, x):
        pred_inter = (x.T / torch.norm(x.T, dim=0)).T
        sqd = torch.cdist(self.class_means[:, :].T, pred_inter)
        return (-sqd).T