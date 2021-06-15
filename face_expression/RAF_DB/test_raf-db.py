import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import cv2
import pandas as pd
import os, torch
import image_utils
import argparse, random
import Networks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='./datasets/FER_2013/', help='Raf-DB dataset path.')
    parser.add_argument('--checkpoint', type=str, default='./models/FER2013/best_model_res.pth', help='Pytorch checkpoint file path')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    return parser.parse_args()
    
    
class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform=None, basic_aug=False):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_test_label.txt'), sep=' ', header=None)
        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            # dataset = df[df[NAME_COLUMN].str.startswith('test')]
            dataset = df
        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.label = dataset.iloc[:, LABEL_COLUMN].values - 1   # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        
        self.file_paths = []
        # use raf-db aligned images for training/testing
        for f in file_names:
            # f = f.split(".")[0]
            # f = f +"_aligned.jpg"
            # path = os.path.join(self.raf_path, 'Image/aligned', f)
            path = os.path.join(self.raf_path, 'Image/', f)
            self.file_paths.append(path)
        
        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image, image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        try:
            image = image[:, :, ::-1]            # BGR to RGB
            label = self.label[idx]
            if self.phase == 'train':
                if self.basic_aug and random.uniform(0, 1) > 0.5:
                    index = random.randint(0, 1)
                    image = self.aug_func[index](image)

            if self.transform is not None:
                image = self.transform(image)
        except:
            print('do nothing')
        
        return image, label, idx, path

        
def run_testing():
    args = parse_args()

    model = Networks.ResNet18()
            
    print("Loading checkpoint weights...", args.checkpoint) 
    checkpoint = torch.load(args.checkpoint)
    model_state_dict = model.state_dict()
    model.load_state_dict(checkpoint["model_state_dict"],strict=False)
        
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_dataset = RafDataSet(args.raf_path, phase='test', transform=data_transforms_val)
    val_num = val_dataset.__len__()
    print('Validation set size:', val_num)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=256,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)
    
    
    model = model.cuda()
    model.eval()
    iter_cnt = 0
    bingo_cnt = 0
    predict_list = torch.Tensor(0).cuda()
    target_list = torch.Tensor(0).cuda()
    path_list = []
    
    for img_i, (imgs, targets, _, paths) in enumerate(val_loader):
        outputs, _ = model(imgs.cuda())
        targets = targets.cuda()
        for path in paths:
          print(path)
        iter_cnt += 1
        _, predicts = torch.max(outputs, 1)
        predict_list = torch.cat((predict_list, predicts), 0)
        target_list = torch.cat((target_list, targets), 0)
        path_list.append(paths)
        correct_or_not = torch.eq(predicts, targets)
        bingo_cnt += correct_or_not.sum().cpu()
        
    acc = bingo_cnt.float()/float(val_num)
    acc = np.around(acc.numpy(), 4)
    print("Validation accuracy:%.4f." % (acc))
    
    predict_np = predict_list.cpu().numpy()
    predict_df = pd.DataFrame(predict_np)
    target_np = target_list.cpu().numpy()
    target_df = pd.DataFrame(target_np)
    predict_df.to_csv('models/predict_fer_res.csv', header=False)
    target_df.to_csv('models/target_fer-res.csv', header=False)
            
if __name__ == "__main__":  
    run_testing()
