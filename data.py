from utils import *
from config import *
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from sklearn.preprocessing import MultiLabelBinarizer
from imgaug import augmenters as iaa
import random
import cv2

# set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)

# create dataset class
class HumanDataset(Dataset):
    def __init__(self,images_df,base_path,augument=True,mode="train"):
        self.images_df = images_df.copy()
        self.augument = augument
        self.images_df.Id = self.images_df.Id.apply(lambda x:base_path + '/' + x)
        self.mlb = MultiLabelBinarizer(classes = np.arange(0,config.num_classes))
        self.mlb.fit(np.arange(0,config.num_classes))
        self.mode = mode

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self,index):
        row = self.images_df.iloc[index]
        filename = str(row.Id)
        X = self.read_images(index)
        if not self.mode == "test":
            if self.images_df.iloc[index].Target == "":
                labels = []
            else:
                labels = np.array(list(map(int, self.images_df.iloc[index].Target.split(' '))))
            y  = np.eye(config.num_classes,dtype=np.float)[labels].sum(axis=0)
        else:
            y = str(self.images_df.iloc[index].Id)
        if not self.mode == "test":
            X = self.new_augumentor(X, int(filename[-1]))
            X = T.Compose([T.ToPILImage(),T.ToTensor()])(X)
            return X.float(),y
        else:
            # TTA
            res = None
            for i in range(2):
                for j in range(4):
                    temp = self.TTA(X, i, j)
                    temp = T.Compose([T.ToPILImage(),T.ToTensor()])(temp).float()
                    if i == 0:
                        res = temp.unsqueeze(0)
                    else:
                        res = torch.cat((res, temp.unsqueeze(0)), 0)
            return res, y

    def Normalize(self, nn, max_val):
        nn_max = nn.max()
        if not nn_max == 0:
            nn = nn / nn_max * max_val
        return nn
    
    def read_images(self,index):
        row = self.images_df.iloc[index]
        filename = str(row.Id)
        filename = filename[:len(filename)-1]
        #use only rgb channels
        if config.channels == 4:
            images = np.zeros(shape=(512,512,4))
        else:
            images = np.zeros(shape=(512,512,3))
        filename_len = len(filename)
        r = np.array(Image.open(filename[:filename_len-4]+"_red" + filename[filename_len-4:]))
        g = np.array(Image.open(filename[:filename_len-4]+"_green" + filename[filename_len-4:]))
        b = np.array(Image.open(filename[:filename_len-4]+"_blue" + filename[filename_len-4:]))
        y = np.array(Image.open(filename[:filename_len-4]+"_yellow" + filename[filename_len-4:]))
        # Normalize
        r = self.Normalize(r, 255)
        g = self.Normalize(g, 255)
        b = self.Normalize(b, 255)
        y = self.Normalize(y, 255)
        images[:,:,0] = r.astype(np.uint8) 
        images[:,:,1] = g.astype(np.uint8)
        images[:,:,2] = b.astype(np.uint8)
        if config.channels == 4:
            images[:,:,3] = y.astype(np.uint8)

        images = images.astype(np.uint8)
        if config.img_height == 512:
            return images
        else:
            return cv2.resize(images,(config.img_weight,config.img_height))
    
    def new_augumentor(self, image, aug_type):
        if aug_type % 2 == 0:
            image = self.TTA1(image)
        if aug_type // 2 == 1:
            image = self.TTA4(image)
        if aug_type // 2 == 2:
            image = self.TTA5(image)
        if aug_type // 2 == 3:
            image = self.TTA6(image)
        return image
    
    def TTA(self, image, n_i, n_j):
        if n_i == 1:
            image = self.TTA1(image)
        if n_j == 1:
            image = self.TTA4(image)
        if n_j == 2:
            image = self.TTA5(image)
        if n_j == 3:
            image = self.TTA6(image)
        return image
    
    def TTA1(self,image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Flipud(1),
            ])], random_order=True)
        
        image_aug = augment_img.augment_image(image)
        return image_aug
    
    def TTA4(self,image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=90),
            ])], random_order=True)
        
        image_aug = augment_img.augment_image(image)
        return image_aug
    
    def TTA5(self,image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=180),
            ])], random_order=True)
        
        image_aug = augment_img.augment_image(image)
        return image_aug
    
    def TTA6(self,image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=270),
            ])], random_order=True)
        
        image_aug = augment_img.augment_image(image)
        return image_aug