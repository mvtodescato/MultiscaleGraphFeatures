import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, KFold, learning_curve
import torch.nn as nn
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader , random_split
from PIL import Image
import clip
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
import sys
sys.path.append('./CLIP')  # git clone https://github.com/openai/CLIP

#loading stanford cars dataset info
from scipy.io import loadmat
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
    
    
df_train = pd.DataFrame(bboxes, columns = ['Filename','bbox_x1', 'bbox_x2', 'bbox_y1', 'bbox_y2','car_class'])
print(df_train)

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
    bboxes.append((fname,bbox_x1, bbox_x2, bbox_y1, bbox_y2, int(list(car_class)[0])))
    
    
df_test = pd.DataFrame(bboxes, columns = ['Filename','bbox_x1', 'bbox_x2', 'bbox_y1', 'bbox_y2','car_class'])
print(df_test)

df_all = pd.concat([df_train,df_test])
print(df_all)


LABELS_MAP_name = loadmat('stanford/cars_annos.mat')

print(LABELS_MAP_name)
ann = LABELS_MAP_name['class_names']
ann = np.transpose(ann)

LABELS_MAP = []

for name in ann:
    LABELS_MAP.append(name[0][0])


target_col = 'car_class'  # Column with labels
input_resolution = 224    # Model input resolution

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
# Definel "classic" models from torch-hub without pre train
def load_torch_hub_model(model_name):
    # Load model
    model = torch.hub.load('pytorch/vision:v0.6.0',
                           model_name, pretrained=False)

    model = model.to(device)



    return model, torch_hub_normalization()


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
            image = self.preprocess(Image.open(row['Filename']))
        except:
            image = self.empty_image

        return image, row[target_col] -1


# %%
# Define model loaders
MODELS_LOADERS = {
    # 'mobilenet': load_mobilenet,   
    'densenet': load_densenet,
    # 'resnet': load_resnet,
    #'resnext': load_resnext,
    # 'vgg': load_vgg,
    #'clip_vit_b': load_clip_vit_b,
    # 'clip_rn50': load_clip_rn50
}


# %%
# Create dataloader
def dataset_loader(model_loader):
    model, image_normalization = model_loader()

    preprocess = Compose([preprocess_image, image_normalization])

    # Create DataLoader with preprocess information related to the model used
    ds = ImagesDataset(df_all, preprocess, input_resolution)

    return ds

# %% [markdown]
# # ☕️
# Coffee time!


# %%
# Models evaluations (may take a while to run this cell)
results = {}
results_acc = {}

# Define folds
folds = list(KFold(5, shuffle=True, random_state=42).split(df_all))

#Final classifier included in the original model
class BaseNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1000, len(LABELS_MAP))
        )

    def forward(self, x):
        return self.network(x)

nn_model = BaseNeuralNetwork()

print("Classifier topology")
print(nn_model)
print()

EPOCHS = 100

print("Epochs", EPOCHS)

prediction_df = []

#Execute all the process to each model in MODELS_LOADERS
n = len(MODELS_LOADERS)
for i, (model_name, model_loader) in enumerate(MODELS_LOADERS.items(), 1):
    print(f'[{i}/{n}] Evaluating on {model_name}...')

    ds = dataset_loader(model_loader)   #load dataset
    model, im = model_loader()
    
    final_matrix = np.zeros((len(LABELS_MAP), len(LABELS_MAP))).astype(int)

    final_test = []
    final_pred = []

    #Train and test each fold
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        #Spliting train and validation sets
        np.random.shuffle(train_idx)
        val = int(len(train_idx) * 0.9)
        training, valid = train_idx[:val], train_idx[val:]
        
        #Concat the original model and the classifier
        net = BaseNeuralNetwork().to(torch_device)
        fold_model = nn.Sequential(model, net)
        
        #Training parameters
        test_df = df_all.iloc[test_idx]
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
        #fold_model.load_state_dict(torch.load(f"stanford_densenet-fold-{fold_idx+1}.model"))
        #fold_model.eval()

        for epoch in range(EPOCHS):  # loop over the dataset multiple times
            running_loss = 0.0

            train_loader = DataLoader(ds, batch_size=16, shuffle=False,
                    num_workers=0, pin_memory=True, sampler = training)

            valid_loader = DataLoader(ds, batch_size=16, shuffle=False,
                    num_workers=0, pin_memory=True, sampler = valid)

            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, label_index = data

                multilabel_values = np.zeros((len(label_index),len(LABELS_MAP))).astype(float)

                for k, idx in enumerate(label_index):
                    multilabel_values[k][idx] = 1.0

                tensor_multilabel_values = torch.from_numpy(multilabel_values).to(torch_device)

                # zero the parameter gradients
                fold_optimizer.zero_grad()

                # forward + backward + optimize
                
                outputs = fold_model(inputs.to(torch_device))
                pred = outputs.cpu().argmax()
                fold_loss = fold_criterion(outputs, tensor_multilabel_values.float())
                fold_loss.backward()
                fold_optimizer.step()

                # print statistics
                running_loss += fold_loss.item()
                
                if i == len(train_loader) - 1:    # print every 2000 mini-batches
                    # print('[%d, %5d] loss: %.3f' %
                    #     (epoch + 1, i + 1, running_loss / 2000))
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

                outputs = fold_model(inputs.to(torch_device))
                pred = outputs.cpu().argmax()

                fold_loss = fold_criterion(outputs, tensor_multilabel_values.float())
                valid_loss += fold_loss.item()
                current_loss = valid_loss / len(valid_loader)
                if i == len(valid_loader) - 1:   
                    print('[%d, %5d] Valid loss: %.5f' %
                        (epoch + 1, i + 1, valid_loss / len(valid_loader)))
                    valid_loss = 0.0

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
                print('trigger times: 0')
                trigger_times = 0

            last_loss = current_loss


        #Plot the learning curve before test
        plt.plot(np.array(learning_c_valid), 'r', label = "valid loss")
        plt.plot(np.array(learning_c_train), 'b', label = "train loss")
        plt.legend()
        plt.savefig(str(fold_idx) + "_densenet_stanford_lc.jpg")
        plt.clf()

        corrects = 0
        fold_cm = np.zeros((len(LABELS_MAP), len(LABELS_MAP))).astype(int)
        y_pred = []
        test_predictions = []

        print("Start testing ...")
        test_y = []
        for i, data in enumerate(ds, 0):
            if i not in test_idx:
                continue
            x_item, y_item = data
            test_y.append(y_item)
            x_item = x_item[None,:]

            item_input = x_item.to(torch_device)

            preds = fold_model(item_input)

            pred_index = preds.cpu().argmax()

            fold_cm[y_item][pred_index] += 1

            if pred_index == y_item:
                corrects += 1

            y_pred.append(pred_index)
            test_predictions.append(preds.detach().cpu().numpy().tolist())

        test_y = np.array(test_y)
        y_pred = np.array(y_pred)
        accuracy_score = corrects / len(test_idx)

        fold_predictions = list(zip(test_df['Filename'].values, test_df['car_class'].values, test_predictions))
        fold_predictions = [[p[0], p[1], *p[2]] for p in fold_predictions]

        prediction_df += fold_predictions        
        
        print(f"{corrects}/{len(test_y)} = val_acc {accuracy_score:.5f}")
        print('Finished fold training')

        final_matrix = np.add(final_matrix, fold_cm)
        final_test = np.concatenate((final_test, test_y))
        final_pred = np.concatenate((final_pred, y_pred)) 

        print("Raw matrix:")
        print(fold_cm.tolist())
        print()

        print("Classification Report:")
        print(metrics.classification_report(test_y, y_pred, target_names=LABELS_MAP))
        print()

        #Saving the classifier model
        torch.save(fold_model.state_dict(), f"stanford_densenet-fold-{fold_idx+1}.model")

    print("Final Result")
    print("Final Raw matrix:")
    print(final_matrix.tolist())
    print()

    print("Final Classification Report:")
    cr = metrics.classification_report(final_test, final_pred, target_names=LABELS_MAP,output_dict=True)
    print()
    df_report = pd.DataFrame(cr).transpose()

    df_report.to_csv('results_densenet_stanford.csv')