import torch
from torchvision import transforms

import numpy as np
import pandas as pd
from PIL import Image

import os
import yaml
from itertools import product

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, random_split
from sklearn.model_selection import train_test_split


def _load_dataset_paths(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


_DATASET_PATHS = _load_dataset_paths("datasets.yaml")


class GroupDataset:
    def __init__(
        self, df, subsample_what=None, duplicates=None
    ):
        self.i = list(range(len(df)))
        self.ids = df["ids"]

        self.y = df["y"].tolist()
        self.g = df["a"].tolist()

        self.count_groups()

        if subsample_what is not None:
            self.subsample_(subsample_what)

        if duplicates is not None:
            self.duplicate_(duplicates)

    def count_groups(self):
        self.wg, self.wy = [], []

        all_groups = sorted(list(set(self.g)))
        all_labels = sorted(list(set(self.y)))
        self.nb_groups = len(all_groups) 
                                        
        self.nb_labels = len(all_labels)
        self.gid = {
            (y, g): i 
            for i, (g, y) in enumerate(product(all_groups, all_labels))
        }
        self.group_sizes = [0] * self.nb_groups * self.nb_labels
        self.class_sizes = [0] * self.nb_labels

        for i in self.i:
            self.group_sizes[self.gid[(self.y[i], self.g[i])]] += 1
            self.class_sizes[self.y[i]] += 1

        for i in self.i:
            self.wg.append(
                len(self) / self.group_sizes[self.gid[(self.y[i], self.g[i])]]
            )
            self.wy.append(len(self) / self.class_sizes[self.y[i]])

    def subsample_(self, subsample_what):
        perm = torch.randperm(len(self)).tolist()

        if subsample_what == "groups":
            min_size = min(list(self.group_sizes))
        else:
            min_size = min(list(self.class_sizes))

        counts_g = [0] * self.nb_groups * self.nb_labels
        counts_y = [0] * self.nb_labels
        new_i = []
        for p in perm:
            y, g = self.y[self.i[p]], self.g[self.i[p]]

            if (
                subsample_what == "groups"
                and counts_g[self.gid[(int(y), int(g))]] < min_size
            ) or (subsample_what == "classes" and counts_y[int(y)] < min_size):
                counts_g[self.gid[(int(y), int(g))]] += 1
                counts_y[int(y)] += 1
                new_i.append(self.i[p])

        self.i = new_i
        self.count_groups()

    def duplicate_(self, duplicates):
        new_i = []
        for i, duplicate in zip(self.i, duplicates):
            new_i += [i] * duplicate
        self.i = new_i
        self.count_groups()

    def __getitem__(self, i):
        j = self.i[i]
        x = self.get_image_by_id(self.ids[j]) 
        y = torch.tensor(self.y[j], dtype=torch.long)
        g = torch.tensor(self.g[j], dtype=torch.long)
        return torch.tensor(i, dtype=torch.long), x, y, g

    def __len__(self):
        return len(self.i)


class FolderGroupDataset(GroupDataset):
    def __init__(self, folder, transform, metadata, subsample_what=None, duplicates=None):
        super().__init__(metadata, subsample_what=subsample_what, duplicates=duplicates)
        self.transform = transform
        self.folder = folder

    def get_image_by_id(self, id):
        return self.transform(Image.open(os.path.join(self.folder, str(id))).convert("RGB"))

class AdultDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.i = list(range(len(df)))
        self.data = df[:,:-2]

        self.y = df[:,-2].tolist()
        self.g = df[:,-1].tolist()

    def __getitem__(self, i):
        j = self.i[i]
        return torch.tensor(i, dtype=torch.long), torch.tensor(self.data[j], dtype=torch.float32), torch.tensor(self.y[j], dtype=torch.long), torch.tensor(self.g[j], dtype=torch.long)

    def __len__(self):
        return len(self.data)
    
class CompasDataset(Dataset):
    def __init__(self, X, y, sensitive_attr):
        self.X = X
        self.y = y
        self.g = sensitive_attr
        self.i = list(range(len(y)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        j = self.i[idx]
        return torch.tensor(idx, dtype=torch.long), self.X[j], self.y[j].long(), self.g[j].long()

def _get_image_train_transform(crop, resize):
    return transforms.Compose(
        [
            transforms.CenterCrop(crop),
            transforms.Resize(resize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

def _get_image_test_transform(crop, resize):
    return transforms.Compose(
        [
            transforms.CenterCrop(crop),
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

def _get_celeba_dataset(transform, split, target='Attractive', sensitive='Male'):
    root = _DATASET_PATHS["celeba"]["root"]
    folder = os.path.join(root, "img_align_celeba/")
    metadata = pd.read_csv(os.path.join(root, _DATASET_PATHS["celeba"]["metadata"]))
    metadata = metadata[metadata['split'] == {"tr": 0, "va": 1, "te": 2}[split]]

    metadata = pd.DataFrame(
        {
            'a': np.array(metadata[sensitive], dtype=int),
            'y': np.array(metadata[target], dtype=int),
            'ids': np.array(metadata['file_name'], dtype=str),
        }
    )
    return FolderGroupDataset(folder, transform, metadata, subsample_what=None, duplicates=None)

def _get_adult_dataset():
    full_data = pd.read_csv(
        _DATASET_PATHS["adult"]["fulldata"],
        names=[
            "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"],
            sep=r'\s*,\s*',
            engine='python', skiprows=1,
            na_values="?", dtype={0:int, 1:str, 2:int, 3:str, 4:int, 5: str, 6:str , 7:str ,8:str ,9: str, 10:int, 11:int, 12:int, 13:str,14: str}
        )

    str_list=[]

    for data in [full_data]:
        for colname, colvalue in data.items(): 
            if type(colvalue[1]) == str:
                str_list.append(colname) 
    num_list = data.columns.difference(str_list)

    for data in [full_data]:
        for i in full_data:
            data[i].replace('nan', np.nan, inplace=True)
        data.dropna(inplace=True)
    
    full_labels = full_data['Target'].copy()
    full_sensitive = full_data['Sex'].copy()
    full_data = full_data.drop(['Target'], axis=1)
    full_data = full_data.drop(['Sex'], axis=1)

    label_encoder1 = LabelEncoder()
    full_labels = label_encoder1.fit_transform(full_labels)
    label_encoder2 = LabelEncoder()
    full_sensitive = label_encoder2.fit_transform(full_sensitive)

    cat_data = full_data.select_dtypes(include=['object']).copy()
    other_data = full_data.select_dtypes(include=['int']).copy()

    columns_to_scale = ['Age', 'Capital Gain', 'Capital Loss', 'Hours per week']
    mms = MinMaxScaler()
    min_max_scaled_columns = mms.fit_transform(other_data[columns_to_scale])

    other_data['Age'],other_data['Capital Gain'],other_data['Capital Loss'],other_data['Hours per week']=\
        min_max_scaled_columns[:,0],min_max_scaled_columns[:,1], min_max_scaled_columns[:,2],min_max_scaled_columns[:,3]
    newcat_data = pd.get_dummies(cat_data, columns=[
        "Workclass", "Education", "Country" ,"Relationship", "Martial Status", "Occupation", "Relationship",
        "Race"], dtype=int)
    full_data = pd.concat([other_data, newcat_data], axis=1)
    full_dataset = np.asarray(full_data).astype(np.float32)
    num_features = full_dataset.shape[1]

    res = np.concatenate([full_dataset, np.array([full_labels]).T, np.array([full_sensitive]).T ], 1)
    np.random.seed(42)
    np.random.shuffle(res)
    train_size = 30000
    valid_size = 5000
    test_size = len(res)- train_size- valid_size
    res_tr = res[:train_size]
    res_va = res[train_size:train_size+valid_size]
    res_te = res[train_size+valid_size:]
    return AdultDataset(res_tr), AdultDataset(res_va), AdultDataset(res_te)

def _get_compas_dataset():
    col_names = [
        "sex",
        "age",
        "race",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
        "c_charge_degree",
        "two_year_recid",
    ]

    data = (
        pd.read_csv("./compas_dataset/compas-scores-two-years.csv")
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )

    data = data.loc[:, col_names]

    data = data.replace(" ?", pd.NA)
    data = data.dropna()

    sensitive_attr = data["race"].apply(lambda x: "African-American" in x).values

    y = LabelEncoder().fit_transform(data["two_year_recid"])

    X = data.drop(["race", "two_year_recid"], axis=1)

    num_cols = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    X = pd.get_dummies(X).values
    X = X.astype("float32")
    y = y.astype("float32")
    sensitive_attr = sensitive_attr.astype("float32")

    (
        X_train,
        X_temp,
        y_train,
        y_temp,
        sensitive_attr_train,
        sensitive_attr_temp,
    ) = train_test_split(X, y, sensitive_attr, test_size=0.4, random_state=42)

    (
        X_val,
        X_test,
        y_val,
        y_test,
        sensitive_attr_val,
        sensitive_attr_test,
    ) = train_test_split(
        X_temp, y_temp, sensitive_attr_temp, test_size=0.5, random_state=42
    )

    reshape_and_tensor = lambda arr: torch.from_numpy(arr).reshape(-1, 1)
    if len(X_train.shape) > 1:
        X_train, X_val, X_test = map(torch.from_numpy, (X_train, X_val, X_test))
    else:
        X_train, X_val, X_test = map(reshape_and_tensor, (X_train, X_val, X_test))
    y_train, y_val, y_test = map(reshape_and_tensor, (y_train, y_val, y_test))
    sensitive_attr_train, sensitive_attr_val, sensitive_attr_test = map(
        reshape_and_tensor,
        (sensitive_attr_train, sensitive_attr_val, sensitive_attr_test),
    )

    train_dataset = CompasDataset(X_train, y_train.reshape(-1), sensitive_attr_train.reshape(-1))
    val_dataset = CompasDataset(X_val, y_val.reshape(-1), sensitive_attr_val.reshape(-1))
    test_dataset = CompasDataset(X_test, y_test.reshape(-1), sensitive_attr_test.reshape(-1))
    return train_dataset, val_dataset, test_dataset

def get_loaders(args, weights=None):
    datasets = {}

    if args['dataset'] == "celeba":
        for split in ['tr', 'va', 'te']:
            transform = {
                'tr': _get_image_train_transform,
                'va': _get_image_test_transform,
                'te': _get_image_test_transform
            }[split](args['crop'], args['resize'])
            datasets[split] = _get_celeba_dataset(
                transform, split,
                target=args['target'],
                sensitive=args['sensitive']
            )
    elif args['dataset'] == 'adult':
        tr, va, te = _get_adult_dataset()

        datasets['tr'] = tr
        datasets['va'] = va
        datasets['te'] = te
    elif args['dataset'] == 'compas':
        tr, va, te = _get_compas_dataset()
        datasets['tr'] = tr
        datasets['va'] = va
        datasets['te'] = te
    else:
        raise ValueError("Unknown dataset: {}".format(args['dataset']))

    loaders = {}
    for split, dataset in datasets.items():
        if split in ['tr', 'va', 'te']:
            if split != 'tr' or weights is None:
                sampler = None
                shuffle = (split == "tr")

            else:
                sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
                shuffle = False

            loaders[split] = torch.utils.data.DataLoader(
                dataset,
                batch_size=args['batch_size'],
                shuffle=shuffle,
                sampler=sampler,
                num_workers=args['num_workers']
            )

    return loaders


