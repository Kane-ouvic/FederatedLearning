import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

os.environ['CUDA_VISIBLE_DEVICES']='-1'

class CellDataset(Dataset):
    def __init__(self, filenames, labels):
        self.filenames = filenames
        self.labels = labels
 
    def __len__(self):
        return len(self.filenames)
 
    def __getitem__(self, idx):
        image = self.filenames[idx]
        label = np.array(self.labels[idx])

        return image, label

def loadingData(imgpath, mode = "image"):
    imglist = []
    images = os.listdir(imgpath)
    for i in images:
        if mode == "image":
            # 如果是RGB影像
            img = Image.open(imgpath + i).convert("RGB")
            img = np.array(img) / 255
            img = img.transpose((2,0,1))

            # 如果是灰階影像
            img = Image.open(imgpath + i).convert("L")
            img = np.array(img) / 255
            img = np.expand_dims(img, axis=0)
        else:
            img = Image.open(imgpath + i).convert("L")
            img = np.array(img) / 255
            img = np.expand_dims(img, axis=0)
        
        imglist.append(img)
   
    return imglist

def split_Train_Val_Data(imgpath, maskpath, batch_size=16):

    train_list = loadingData(imgpath, mode = "image")
    mask_list = loadingData(maskpath, mode = "label")

    # ----------------------------------
    # split Train / validation with 8:2 
    # ----------------------------------
       
    num_sample_train = int(0.8 * len(train_list))
    num_sample_val = len(train_list) - num_sample_train
        
    # print(str(len(train_list)) + ' | ' + str(num_sample_train) + ' | ' + str(num_sample_val))

    train_inputs = train_list[:num_sample_train]
    train_labels = mask_list[:num_sample_train]

    val_inputs = train_list[num_sample_train:]
    val_labels = mask_list[num_sample_train:]

    train_dataloader = DataLoader(CellDataset(train_inputs, train_labels), batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(CellDataset(val_inputs, val_labels), batch_size = batch_size, shuffle = False)
 
    return train_dataloader, val_dataloader

if __name__ == '__main__':
    imgpath = './images/train/images/'
    maskpath = './images/train/label/'
    split_Train_Val_Data(imgpath, maskpath)
