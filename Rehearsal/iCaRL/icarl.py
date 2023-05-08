
from math import ceil
import numpy as np

import torch
from torch import nn

class ICaRLPlugin(nn.Module):
    def __init__(self, args, classes_order, model, device, buffer_transform=None, fixed_memory=True):
        super().__init__()

        self.memory_size = args.memory
        self.buffer_transform = buffer_transform
        self.fixed_memory = fixed_memory
        self.classes_order = classes_order
        self.x_memory = []
        self.y_memory = []
        self.order = []
        self.model = model
        self.old_model = None
        self.observed_classes = []
        self.embedding_size = None

        #####
        self.args = args
        self.class_means = None
        self.device = device

    def before_training_exp(self, prev_idx):
        nb_cl = self.args.class_increment
        previous_seen_classes = prev_idx
        self.observed_classes.extend(self.classes_order[previous_seen_classes : previous_seen_classes + nb_cl])

    def before_forward(self, feature_size):
        with torch.no_grad():
            self.embedding_size = feature_size

    def after_training_exp(self, train_loader):
        self.model.eval()
        print('after_training_exp')

        self.compute_class_means(train_loader)
        # self.construct_exemplar_set(device)
        # self.reduce_exemplar_set()

    def compute_class_means(self, train_loader):
        if self.class_means is None:
            n_classes = int(self.args.num_classes / self.args.class_increment)
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
                self.exemplar_means.append(cls_feats_mean)
                
                label = self.y_memory[i][0]
                class_samples = class_samples.to(self.device)

                mapped_prototypes = self.model.feature_extractor(class_samples).detach()
                D = mapped_prototypes.T
                D = D / torch.norm(D, dim=0)

                if len(class_samples.shape) == 4:
                    class_samples = torch.flip(class_samples, [3])

                with torch.no_grad():
                    mapped_prototypes2 = self.model.feature_extractor(class_samples).detach()

                D2 = mapped_prototypes2.T
                D2 = D2 / torch.norm(D2, dim=0)

                div = torch.ones(class_samples.shape[0], device=self.device)
                div = div / class_samples.shape[0]

                m1 = torch.mm(D, div.unsqueeze(1)).squeeze(1)
                m2 = torch.mm(D2, div.unsqueeze(1)).squeeze(1)
                self.class_means[:, label] = (m1 + m2) / 2
                self.class_means[:, label] /= torch.norm(self.class_means[:, label])

    def construct_exemplar_set(self, device):
        nb_cl = self.args.class_increment
        previous_seen_classes = len(self.observed_classes)

        if self.fixed_memory:
            nb_protos_cl = int(ceil(self.memory_size / len(self.observed_classes)))
        else:
            nb_protos_cl = self.memory_size
        
        new_classes = self.observed_classes[previous_seen_classes : previous_seen_classes + nb_cl]

        dataset = dataset
        targets = torch.tensor(dataset.targets)
        
        for iter_dico in range(nb_cl):
            cd = classification_subset(dataset, torch.where(targets == new_classes[iter_dico])[0])
            collate_fn = cd.collate_fn if hasattr(cd, "collate_fn") else None

            eval_dataloader = DataLoader(cd.eval(), collate_fn=collate_fn,batch_size=eval_mb_size)

            class_patterns = []
            mapped_prototypes = []
            for idx, (class_pt, _, _) in enumerate(eval_dataloader):
                class_pt = class_pt.to(device)
                class_patterns.append(class_pt)
                with torch.no_grad():
                    mapped_pttp = (self.model.feature_extractor(class_pt).detach())
                mapped_prototypes.append(mapped_pttp)

            class_patterns = torch.cat(class_patterns, dim=0)
            mapped_prototypes = torch.cat(mapped_prototypes, dim=0)

            D = mapped_prototypes.T
            D = D / torch.norm(D, dim=0)

            mu = torch.mean(D, dim=1)
            order = torch.zeros(class_patterns.shape[0])
            w_t = mu

            i, added, selected = 0, 0, []
            while not added == nb_protos_cl and i < 1000:
                tmp_t = torch.mm(w_t.unsqueeze(0), D)
                ind_max = torch.argmax(tmp_t)

                if ind_max not in selected:
                    order[ind_max] = 1 + added
                    added += 1
                    selected.append(ind_max.item())

                w_t = w_t + mu - D[:, ind_max]
                i += 1

            pick = (order > 0) * (order < nb_protos_cl + 1) * 1.0
            self.x_memory.append(class_patterns[torch.where(pick == 1)[0]])
            self.y_memory.append([new_classes[iter_dico]] * len(torch.where(pick == 1)[0]))
            self.order.append(order[torch.where(pick == 1)[0]])

    def reduce_exemplar_set(self):
        tid = clock.train_exp_counter
        nb_cl = experience.benchmark.n_classes_per_exp

        if self.fixed_memory:
            nb_protos_cl = int(ceil(self.memory_size / len(self.observed_classes)))
        else:
            nb_protos_cl = self.memory_size

        for i in range(len(self.x_memory) - nb_cl[tid]):
            pick = (self.order[i] < nb_protos_cl + 1) * 1.0
            self.x_memory[i] = self.x_memory[i][torch.where(pick == 1)[0]]
            self.y_memory[i] = self.y_memory[i][: len(torch.where(pick == 1)[0])]
            self.order[i] = self.order[i][torch.where(pick == 1)[0]]

    

    def forward(self, x):
        pred_inter = (x.T / torch.norm(x.T, dim=0)).T
        sqd = torch.cdist(self.class_means[:, :].T, pred_inter)
        return (-sqd).T