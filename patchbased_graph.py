import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, KFold
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import networkx as nx
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import Dataset
from PIL import Image
import clip
from collections import Counter
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from numpy import asarray, dtype
import math
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import TopKPooling

import getopt, sys
from scipy.io import loadmat
sys.path.append('./CLIP') 


target_col = 'class_label'  # Default column with labels
input_resolution = 224    # Default input resolution


# %%
torch_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# %%
def to_rgb(image):
    return image.convert("RGB")


# General transformation applied to all models
preprocess_image = Compose(
    [
        Resize(input_resolution, interpolation=Image.BICUBIC),
        CenterCrop(input_resolution),
        to_rgb,
        ToTensor(),
    ]
)


# %%
def torch_hub_normalization():
    # Normalization for torch hub vision models
    return Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    )


# %%
def clip_normalization():
    # SRC https://github.com/openai/CLIP/blob/e5347713f46ab8121aa81e610a68ea1d263b91b7/clip/clip.py#L73
    return Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )


# %%
# Definel "classic" models from torch-hub
def load_torch_hub_model(model_name):
    # Load model
    model = torch.hub.load('pytorch/vision:v0.6.0',
                           model_name, pretrained=True)

    # Put model in 'eval' mode and sent to device
    model = model.eval().to(device)

    # Check for features network
    if hasattr(model, 'features'):
        features = model.features
    else:
        features = model

    return features, torch_hub_normalization()


def load_mobilenet():
    return load_torch_hub_model('mobilenet_v2')


def load_densenet():
    return load_torch_hub_model('densenet121')


def load_resnet():
    return load_torch_hub_model('resnet101')


def load_resnext():
    return load_torch_hub_model('resnext101_32x8d')


def load_vgg():
    return load_torch_hub_model('vgg16')


# %%
# Define CLIP models (ViT-B and RN50)
def load_clip_vit_b():
    model, _ = clip.load("ViT-B/32", device=device)

    return model.encode_image, clip_normalization()


def load_clip_rn50():
    model, _ = clip.load("RN50", device=device)

    return model.encode_image, clip_normalization()


# %%
# Dataset loader
class ImagesDataset(Dataset):
    def __init__(self, df, preprocess, input_resolution):
        super().__init__()
        self.df = df
        self.preprocess = preprocess
        self.empty_image = torch.zeros(3, input_resolution, input_resolution)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        try:
            image = self.preprocess(Image.fromarray(row['Filename']))
        except:
            image = self.empty_image

        return image, row[target_col]


# %%
# Define model loaders
MODELS_LOADERS = {
     'mobilenet': load_mobilenet,
     'densenet': load_densenet,
     'resnet': load_resnet,
     'resnext': load_resnext,
     'vgg': load_vgg,
     'clip_vit_b': load_clip_vit_b,
     'clip_rn50': load_clip_rn50
}




# %% [markdown]
# ## Patches generation

def generate_patchs(im, N, s):
    W = input_resolution  #window width
    H = input_resolution  #window height
    
    s = int(s * W)   # Sliding window stride
    
    tam = N * W
    width, height = im.size #image original dimensions size

    #calculing the new size
    if width < height:
        small = 'w'
        mult = tam/width
        new_w = tam
        new_h = math.ceil(height * mult)
    else:
        small = 'h'
        mult = tam/height
        new_w = math.ceil(width * mult)
        new_h = tam

    orig_im = asarray(im)
    

    #resize image
    transform = T.Resize([int(new_h), int(new_w)], interpolation=Image.BICUBIC)
    im = transform(im)
    im = asarray(im)

    tiles = []
    cord = []
    #patches generation
    for x in range(0,im.shape[0],W):
        for y in range(0,im.shape[1],H):
            tiles.append(im[x:x+W,y:y+H])
            cord.append([x,y])
    

    for strt in range(s,W,s):
        for x in range(strt,im.shape[0],W):
            for y in range(0,im.shape[1],H):
                tiles.append(im[x:x+W,y:y+H])
                cord.append([x,y])

    for strt in range(s,W,s):
        for x in range(0,im.shape[0],W):
            for y in range(strt,im.shape[1],H):
                tiles.append(im[x:x+W,y:y+H])
                cord.append([x,y])



    #drop unsized patches

    final_tiles = []
    final_cord = []
    for index in range(len(tiles)):
        im_test = Image.fromarray(tiles[index])
        width, height = im_test.size    
        if width * height == W * H:
            final_tiles.append(tiles[index])
            final_cord.append(cord[index])

    #ordering patches
    cord_index = final_tiles
    cord_index2 = final_cord

    #creating edges
    edges = []
    for index in range(len(cord_index2)):
        i,j=cord_index2[index]
        for index2 in range(len(cord_index2)):
            if index == index2:
                continue
            i2, j2 = cord_index2[index2]
            if i2 == i or i2+s == i or i2-s == i:
                if j2 == j or j2+s == j or j2-s == j:
                    edges.append([index, index2])

    print("Num Patchs:")
    print(len(cord_index))
    #print(edges)

    if len(final_tiles) == 1:
        tile = []
        tile.append(orig_im)
        edges.append([0,0])
        return tile, edges
    else:
        return cord_index, edges


#open image and send it to patch split
def df_patchs(row,columns, N, s):
    df2 = pd.DataFrame(data=None, columns=columns)
    try:
        image = Image.open(row['Filename'])
    except ValueError:
        image = torch.zeros(3, input_resolution, input_resolution)
        transf = T.ToPILImage()
        image = transf(image)

    patchs, edges = generate_patchs(image, N, s)
    for patch in patchs:
        row['Filename'] = patch
        df2 = df2.append(row,ignore_index=True)


    return df2, edges    






# %%
# Main function to generate features as describe in paper code
def generate_features(model_loader,N, s, df_all):
    
    # Create model and image normalization
    model, image_normalization = model_loader()
    preprocess = Compose([preprocess_image, image_normalization])
    
    # Sample one output from model just to check output_dim
    x = torch.zeros(1, 3, input_resolution, input_resolution, device=device)
    with torch.no_grad():
        x_out = model(x)
    output_dim = x_out.shape[1]

    #Feature vector
    V_X = []
    V_y = []

    #Process each image to create the context features vector
    for index in range(len(df_all)):
        print(f'Image ({index+1}/{len(df_all)})')
        row = df_all.iloc[index]

        new_df, edges = df_patchs(row, df_all.columns, N, s) #function to split image into patches

 
    
        ds = ImagesDataset(new_df, preprocess, input_resolution)
        dl = DataLoader(ds, batch_size=256, shuffle=False,
                        num_workers=0, pin_memory=True)

        # Features data
        X = np.empty((len(ds), output_dim), dtype=np.float32)

        # Begin feature generation
        i = 0
        for images, cls in tqdm(dl):
            
            n_batch = len(images)

            with torch.no_grad():
                emb_images = model(images.to(device))
                if emb_images.ndim == 4:
                    emb_images = emb_images.reshape(
                        n_batch, output_dim, -1).mean(-1)
                emb_images = emb_images.cpu().float().numpy()

            # Save normalized features
            X[i:i+n_batch] = emb_images / \
                np.linalg.norm(emb_images, axis=1, keepdims=True)
            y = cls

            i += n_batch
        

        y = int(y[0])
        #Patches features agregation by average
        X = torch.from_numpy(X)
        edges = torch.tensor(edges, dtype=torch.long)
        data = Data(x=X, edge_index=edges.t().contiguous())

        V_X.append(data)
        V_y.append(y)
        del ds, dl, X
    
    del model, image_normalization,

    n_y = np.asarray(V_y, dtype=np.int32)

    return V_X, n_y

def index_sha1(df):
    '''
    Index dataframe by image sha1 (found by regex)
    '''
    sha1 = df['Filename'].str.replace(
        r'.*sha1+-(\w+)\.\w{3}', r'\1', regex=True)
    sha1.name = 'sha1'
    return df.set_index(sha1)

def load_geo_dataset():
    input_path = Path('')
    images_path = input_path / \
        'C:/Users/mvtod/dataset_classificador/geodigital-dev.inf.ufrgs.br/~eslgastal/dataset/images/by-hash/'
    annotations_path = input_path / 'C:/Users/mvtod/GeoimageClassifier/dataset/annotations/clean/'


    print("input path", input_path)
    print("images path", images_path)
    print("annotations path", annotations_path)

    # %%
    assert images_path.exists(), f'Folder {images_path} does not exists'
    assert annotations_path.exists(), f'Folder {annotations_path} does not exists'

    # %% [markdown]
    # # Data preparation

    # %%
    all_files = [f for f in images_path.rglob('*') if not f.is_dir()]
    print('Found {} files'.format(len(all_files)))

        # %%
    # Most common suffixes (just checking if all files are images)
    cnt_suffix = Counter([f.suffix for f in all_files])
    print('Most common files')
    print(cnt_suffix.most_common())


    df = index_sha1(pd.read_csv(annotations_path /
                            'v14-one-tree.csv'))


    df_all = df

    # ADD GRAPH AND MAP DATASETS
    # # %%
    # # Read graph and map datasets
    df_graph = index_sha1(pd.read_csv(
        annotations_path / 'v14-graph-subtree-classifications-clean--2020-11-04--708ac6b79--25517.csv'))
    df_map = index_sha1(pd.read_csv(
        annotations_path / 'v14-map-subtree-classifications-clean--2020-11-04--708ac6b79--25517.csv'))

    # # Join main, graph and map annotations. Join by sha1
    df_all = df.join(df_graph, how='outer', rsuffix=' Graph').join(
        df_map, how='outer', rsuffix=' Map')


    # # %%
    # # Filename and category will come from most specific class (map or graph) then to more general
    df_all['Filename'] = df_all['Filename Map'].fillna(
        df_all['Filename Graph']).fillna(df_all['Filename'])
    df_all['Category'] = df_all['Category Map'].fillna(
        df_all['Category Graph']).fillna(df_all['Category']).astype('category')

    # # Sanity check
    assert df_all[['Filename', 'Category']].isnull().sum().sum() == 0

    # Sanity check
    assert Path(df_all['Filename'].iloc[0]).exists()


    # %%
    # Load "important" info (tip: not used)
    df_important = index_sha1(pd.read_csv(
        annotations_path / 'v14-important-images--2020-11-04--708ac6b79--25517.csv'))

    df_all['is_important'] = (df_important.Important.reindex(df_all.index) == 'Y')

    # %%
    # Map to more generic classes
    LABELS_MAP = {
        "3d block diagram": "3d block diagram",
        "fluxogram": "fluxogram",
        "seismic section": "seismic section",
        "chromatogram": "chromatogram",
        "radar chart": "radar chart",
        "person portrait photograph": "person portrait photograph",
        "van krevelen diagram": "van krevelen diagram",
        "aerial photograph": "aerial photograph",
        "photomicrograph": "photomicrograph",
        "line graph": "line graph",
        "geological cross section": "geological cross section",
        "outcrop photograph": "outcrop photograph",
        "scanned page": "scanned page",
        "rose diagram": "rose diagram",
        "hand sample photograph": "hand sample photograph",
        "diffractogram": "diffractogram",
        "bar chart": "bar chart",
        "ternary diagram": "ternary diagram",
        "sattelite image": "fotografia",
        "scanning electron microscope image": "scanning electron microscope image",
        "stereogram": "stereogram",
        "geological chart": "geological chart",
        "well core photograph": "well core photograph",
        "box plot": "box plot",
        "profile": "profile",
        "variogram": "variogram",
        "geological map": "geological map",
        "geotectonic map": "geotectonic map",
        "reference map": "reference map",
        "temperature map": "temperature map",
        "oil ship photograph": "oil ship photograph",
        "oil rig photograph": "oil rig photograph",
        "equipment photograph": "equipment photograph",
        "3d visualization": "3d visualization",
        "submarine arrangement": "submarine arrangement",
        "schematic drawing": "schematic drawing",
        "completion scheme": "completion scheme",
        "seismic cube": "seismic cube",
        "geological process sketch": "geological process sketch",
        "microfossil photograph": "microfossil photograph",
        "geophysical map": "geophysical map",
        "stratigraphic isoattribute map": "stratigraphic isoattribute map",
        "structure contour map": "structure contour map",
        "scatter plot": "scatter plot",
        "table": "table",
        "equipment sketch": "equipment sketch",
        "geology sketch": "geology sketch",
        "plant and project sketch": "plant and project sketch",
    }


    # %%
    df_all['class_label'] = df_all['Category'].map(
        LABELS_MAP).astype('category')
    # df_all['class_label'] = df_all['Category'].astype('category')

    df_all = df_all[df_all['class_label'].isin(LABELS_MAP.values())]

    # old_columns = df_all.columns.tolist()

    # for column in old_columns:
    #     if column not in LABELS_MAP:
    #         df_all.drop([column])

    # %% [markdown]
    # # Feature generation using different models

    print("Columns")
    print(df_all.columns.tolist())
    print()

    print("Full DF")
    print(df_all.groupby("class_label").class_label.count().sort_values(ascending=False))
    print()

    categories = list(df_all[target_col].cat.categories)
    cat_codes = dict(zip(categories, range(len(categories))))

    df_all = df_all.replace({"class_label":cat_codes})

    LABELS_MAP = {
        "3d block diagram": "3d block diagram",
        "fluxogram": "fluxogram",
        "seismic section": "seismic section",
        "chromatogram": "chromatogram",
        "radar chart": "radar chart",
        "person portrait photograph": "person portrait photograph",
        "van krevelen diagram": "van krevelen diagram",
        "aerial photograph": "aerial photograph",
        "photomicrograph": "photomicrograph",
        "line graph": "line graph",
        "geological cross section": "geological cross section",
        "outcrop photograph": "outcrop photograph",
        "rose diagram": "rose diagram",
        "hand sample photograph": "hand sample photograph",
        # "map": "map",
        "diffractogram": "diffractogram",
        "bar chart": "bar chart",
        "ternary diagram": "ternary diagram",
        # "graph": "graph",
        "scanning electron microscope image": "scanning electron microscope image",
        "stereogram": "stereogram",
        "geological chart": "geological chart",
        "well core photograph": "well core photograph",
        "box plot": "box plot",
        "profile": "profile",
        "variogram": "variogram",
        "geological map": "geological map",
        "geotectonic map": "geotectonic map",
        "reference map": "reference map",
        "temperature map": "temperature map",
        "oil ship photograph": "oil ship photograph",
        "oil rig photograph": "oil rig photograph",
        "equipment photograph": "equipment photograph",
        "3d visualization": "3d visualization",
        "submarine arrangement": "submarine arrangement",
        "completion scheme": "completion scheme",
        "seismic cube": "seismic cube",
        "geological process sketch": "geological process sketch",
        "microfossil photograph": "microfossil photograph",
        "geophysical map": "geophysical map",
        "stratigraphic isoattribute map": "stratigraphic isoattribute map",
        "structure contour map": "structure contour map",
        "scatter plot": "scatter plot",
        "table": "table",
        "equipment sketch": "equipment sketch",
        "geology sketch": "geology sketch",
        "plant and project sketch": "plant and project sketch",
    }

    return df_all, LABELS_MAP




def load_stanford_dataset():
    annots = loadmat('stanford/cars_train_annos.mat')

    annotations = annots['annotations']
    annotations = np.transpose(annotations)

    fnames = []
    bboxes = []

    for annotation in annotations:
        bbox_x1 = annotation[0][0][0][0]
        bbox_y1 = annotation[0][1][0][0]
        bbox_x2 = annotation[0][2][0][0]
        bbox_y2 = annotation[0][3][0][0]
        fname = 'stanford/cars_train/' + str(annotation[0][5][0])
        car_class = annotation[0][4][0]
        bboxes.append((fname,bbox_x1, bbox_x2, bbox_y1, bbox_y2, int(list(car_class)[0])))
        
        
    df_train = pd.DataFrame(bboxes, columns = ['Filename','bbox_x1', 'bbox_x2', 'bbox_y1', 'bbox_y2','class_label'])

    annots = loadmat('stanford/cars_test_annos_withlabels_eval.mat')

    annotations = annots['annotations']
    annotations = np.transpose(annotations)

    fnames = []
    bboxes = []

    for annotation in annotations:
        bbox_x1 = annotation[0][0][0][0]
        bbox_y1 = annotation[0][1][0][0]
        bbox_x2 = annotation[0][2][0][0]
        bbox_y2 = annotation[0][3][0][0]
        fname = 'stanford/cars_test/' + str(annotation[0][5][0])
        car_class = annotation[0][4][0]
        bboxes.append((fname,bbox_x1, bbox_x2, bbox_y1, bbox_y2, int(list(car_class)[0])-1))
        
        
    df_test = pd.DataFrame(bboxes, columns = ['Filename','bbox_x1', 'bbox_x2', 'bbox_y1', 'bbox_y2','class_label'])

    df_all = pd.concat([df_train,df_test])
    print(df_all)

    LABELS_MAP_name = loadmat('stanford/cars_annos.mat')

    ann = LABELS_MAP_name['class_names']
    ann = np.transpose(ann)

    LABELS_MAP = []

    for name in ann:
        LABELS_MAP.append(name[0][0])

    return df_all, LABELS_MAP


def classifier(features_list, y, out_name, df_all, LABELS_MAP):
    
    X = features_list[0]
    
    #Saving features
    with open(f'Graphs_overlapping_features_{out_name}.pkl', 'wb') as f:
        pickle.dump(X, f)

    np.save(f"y_labels_{out_name}.npy", y)


    class GCN2(torch.nn.Module):
        def __init__(self):
            super(GCN2, self).__init__()
            self.initial_conv = GCNConv(X[0].num_features, 64)
            self.out = nn.Linear(64, len(LABELS_MAP))

        def forward(self,x,edge_index,batch_index):
            hidden = self.initial_conv(x, edge_index)
            hidden = F.tanh(hidden)
            hidden = gmp(hidden, batch_index)
            out = self.out(hidden)
            return out


    # Define folds
    folds = list(KFold(5, shuffle=True, random_state=42).split(df_all))
    with open("folds","wb") as fl:
        pickle.dump(folds, fl)

    EPOCHS = 100

    print("Epochs", EPOCHS)

    prediction_df = []

    print("Context Features generated sucessfully")
    print("Starting the Classifier train/test")
    
    final_matrix = np.zeros((len(LABELS_MAP), len(LABELS_MAP))).astype(int)

    final_test = []
    final_pred = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        
        #Spliting train and validation sets
        np.random.shuffle(train_idx)
        val = int(len(train_idx) * 0.9)
        training, valid = train_idx[:val], train_idx[val:]
        train_X = []
        for index in training:
            train_X.append(X[index])

        train_y = y[training]
        train_y = torch.from_numpy(train_y)

        valid_X = []
        for index in valid:
            valid_X.append(X[index])
        valid_y = y[valid]
        valid_y = torch.from_numpy(valid_y)

        test_X = []
        for index in test_idx:
            test_X.append(X[index])
        test_y = y[test_idx]
        test_y = torch.from_numpy(test_y)

        test_df = df_all.iloc[test_idx]


        fold_model = GCN2()
        fold_model.to(torch_device)

        print(fold_model)
        print()


        #Training criterion/optimizer/parameters
        fold_criterion = nn.CrossEntropyLoss()
        fold_optimizer = torch.optim.Adam(fold_model.parameters(), lr=0.001)
        learning_c_train = []
        learning_c_valid = []
        last_loss = 100
        patience = 5
        trigger_times = 0
        min_improvement = 0.001

        print("Loss", fold_criterion)
        print("Optimizar", fold_optimizer)
        print()

        print("Start training ...")

        for epoch in range(EPOCHS):  # loop over the dataset multiple times

            running_loss = 0.0

            train_loader = DataLoader(list(zip(train_X, train_y)), batch_size=64, shuffle=False,
                        num_workers=0, pin_memory=True)

            valid_loader = DataLoader(list(zip(valid_X, valid_y)), batch_size=64, shuffle=False,
                        num_workers=0, pin_memory=True)

            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                
                inputs, label_index = data

                multilabel_values = np.zeros((len(label_index),len(LABELS_MAP))).astype(float)

                for k, idx in enumerate(label_index):
                    
                    multilabel_values[k][idx] = 1.0


                tensor_multilabel_values = torch.from_numpy(multilabel_values).to(torch_device)
                #print(tensor_multilabel_values.shape)
                # zero the parameter gradients
                fold_optimizer.zero_grad()

                # forward + backward + optimize
                outputs = fold_model(inputs.x.float().to(torch_device), inputs.edge_index.to(torch_device), inputs.batch.to(torch_device))

                pred = outputs.cpu().argmax()
                
                fold_loss = fold_criterion(outputs, tensor_multilabel_values.float())

                fold_loss.backward()
                fold_optimizer.step()

                # print statistics
                running_loss += fold_loss.item()
                
                if i == len(train_loader) - 1:
                    print('[%d, %5d] Train loss: %.5f' %
                        (epoch + 1, i + 1, running_loss / len(train_loader)))
                    learning_c_train.append(running_loss / len(train_loader))
                    running_loss = 0.0
            
            #Validation
            valid_loss = 0.0
            fold_model.eval() 
            for i, data in enumerate(valid_loader, 0):

                inputs, label_index = data
                multilabel_values = np.zeros((len(label_index),len(LABELS_MAP))).astype(float)

                for k, idx in enumerate(label_index):
                    multilabel_values[k][idx] = 1.0


                tensor_multilabel_values = torch.from_numpy(multilabel_values).to(torch_device)

                fold_optimizer.zero_grad()

                outputs = fold_model(inputs.x.float().to(torch_device), inputs.edge_index.to(torch_device), inputs.batch.to(torch_device))
                pred = outputs.cpu().argmax()

                fold_loss = fold_criterion(outputs, tensor_multilabel_values.float())
                valid_loss += fold_loss.item()
                current_loss = valid_loss / len(valid_loader)
                
                #print statistics
                if i == len(valid_loader) - 1:   
                    print('[%d, %5d] Valid loss: %.5f' %
                        (epoch + 1, i + 1, valid_loss / len(valid_loader)))
                    valid_loss = 0.0

            
            #Early stopping verification
            learning_c_valid.append(current_loss)
            minimal = last_loss - (last_loss * min_improvement)
            print("Minimal:")
            print(minimal)

            if current_loss > minimal:
                trigger_times += 1
                print('Trigger Times:', trigger_times)

                if trigger_times >= patience:
                    print('Early stopping!\nStart to test process.')
                    break

            else:
                print('Trigger times: 0')
                trigger_times = 0

            last_loss = current_loss


        #Plot the learning curve before the test
        plt.plot(np.array(learning_c_valid), 'r', label = "valid loss")
        plt.plot(np.array(learning_c_train), 'b', label = "train loss")
        plt.legend()
        plt.savefig(str(fold_idx) + out_name + "_lc.jpg")
        plt.clf()

        corrects = 0
        fold_cm = np.zeros((len(LABELS_MAP), len(LABELS_MAP))).astype(int)
        y_pred = []
        test_predictions = []

        print("Start testing ...")

        for x_item, y_item in list(zip(test_X, test_y)):

            inputs = x_item
            batch = torch.zeros([1], dtype=torch.long).to(torch_device)
            preds = fold_model(inputs.x.float().to(torch_device), inputs.edge_index.to(torch_device), None)

            pred_index = preds.cpu().argmax()
            fold_cm[y_item][pred_index] += 1

            if pred_index == y_item:
                corrects += 1

            y_pred.append(pred_index)
            test_predictions.append(preds.detach().cpu().numpy().tolist())

        
        #Calculatin the metrics
        y_pred = np.array(y_pred)
        accuracy_score = corrects / len(test_y)

        fold_predictions = list(zip(test_df['Filename'].values, test_df['class_label'].values, test_predictions))
        fold_predictions = [[p[0], p[1], *p[2]] for p in fold_predictions]

        prediction_df += fold_predictions        
        
        print(f"{corrects}/{len(test_y)} = val_acc {accuracy_score:.5f}")
        print('Finished fold training')

        final_matrix = np.add(final_matrix, fold_cm)
        final_test = np.concatenate((final_test, test_y))
        final_pred = np.concatenate((final_pred, y_pred)) 

        print("Fold Classification Report:")
        cr = metrics.classification_report(test_y, y_pred, target_names=LABELS_MAP,output_dict=True)
        print(cr)


        #saving the classifier model
        torch.save(fold_model.state_dict(), f"{out_name}-fold-{fold_idx+1}.model")

    print("Final Result")
    print("Final Raw matrix:")
    print(final_matrix.tolist())
    print()

    print("Final Classification Report:")
    cr = metrics.classification_report(final_test, final_pred, target_names=LABELS_MAP,output_dict=True)
    print(cr)
    df_report = pd.DataFrame(cr).transpose()

    df_report.to_csv(out_name + '_results.csv')




def context_features(Nlist, FElist, s):

    # Function to load stanford cars dataset
    # If you want to load another dataset you need to keep this format:
    # - df_all: pandas dataframe with a column 'Filename' with the image file path and a column 'class_labels' with the label value of each image
    # - LABELS_MAP: list with the name of all classes in the dataset (according to dataframe labels)
    df_all, LABELS_MAP = load_stanford_dataset()
    #df_all, LABELS_MAP = load_geo_dataset()
    
    
    n = len(FElist)
    for i, (model_name) in enumerate(FElist, 1):
        model_loader = MODELS_LOADERS[model_name]
        
        print(f'[{i}/{n}] Evaluating on {model_name}...')
        
        out_name = 'Stanford_' + str(model_name) +'_s' + str(s) + '_N' + ('_'.join(str(n) for n in Nlist))
        print(out_name)
        
        features_list = [] 
        for index in range(len(Nlist)):         
            X, y = generate_features(model_loader, int(Nlist[index]), s, df_all)
            features_list.append(X)     
        
        
        classifier(features_list, y, out_name, df_all, LABELS_MAP)



def main():
    argumentList = sys.argv[1:]
    # Options
    options = "hN:s:"
    # Long options
    long_options = ["Help", "Output=", "FE="]
    out_name = ''
    s = 0
    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)
        
        # checking each argument
        for currentArgument, currentValue in arguments:
    
            if currentArgument in ("-h", "--help"):
                print ("Multiscale context features\n")
                print ('Usage:')
                print('python context.py -h | --help')
                print("python context.py -N <approach_parameter> --FE <model_name> [-s <stride_value>]")
                print('\nOptions:')
                print("-h --help    Show this screen")
                print("-N           Approach parameter (If you want use multiscale you can pass a list of N values separed by a comma, ex: 1,2)")
                print("--FE         Feature extractor model name [models available below] (If you want apply multiple models you can pass a list of names separed by a comma, ex: densenet,resnext)")
                print("-s           Slinding window stride value. Range: (0,1) [default: 0.01]")
                print("\nAvailable feature extractors (call by the name in the right): ")
                print("MobileNet_v2:   mobilenet")
                print("DenseNet_121:   densenet")
                print("ResNet_101:     resnet")
                print("ResNeXt_101:    resnext")
                print("Vgg_16:         vgg")
                print("Clip ViT-B/32:  clip_vit_b")
                print("Clip_RN50:      clip_rn50")
                print("\nMore details of the approach and the implementation:")
                print("LINKS")
                sys.exit(2)

            elif currentArgument in ("-N"):
                if ',' in currentValue:
                    Nlist = currentValue.split(',')
                else:
                    Nlist = [currentValue]

            elif currentArgument in ("--FE"):
                if ',' in currentValue:
                    FElist = currentValue.split(',')
                else:
                    FElist = [currentValue]

            elif currentArgument in ("-s"):
                s = float(currentValue)
                
            
        if s == 0:
            s = 0.01
        print("N: ", Nlist)
        print('Feature Extractors: ', FElist)
        print("Sliding window stride: ", s)

    except UnboundLocalError:
        print("You forget to define something")
        sys.exit(2)
    except getopt.error as err:
        print (str(err))
        sys.exit(2)

    context_features(Nlist, FElist, s)
    


if __name__ == "__main__":
   main()
