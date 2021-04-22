from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import json

## Note that: here we provide a basic solution for loading data and transforming data.
## You can directly change it if you find something wrong or not good enough.

## the mean and standard variance of imagenet dataset
## mean_vals = [0.485, 0.456, 0.406]
## std_vals = [0.229, 0.224, 0.225]

def load_data(data_dir = "../data/",input_size = 224,batch_size = 36, task='1-Large-Scale'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    ## The default dir is for the first task of large-scale deep learning
    ## For other tasks, you may need to modify the data dir or even rewrite some part of 'data.py'
    # task = '1-Large-Scale'
    # task = '2-Medium-Scale'
    # task = '4-Long-Tailed'
    image_dataset_train = datasets.ImageFolder(os.path.join(data_dir, task, 'train'), data_transforms['train'])
    image_dataset_valid = datasets.ImageFolder(os.path.join(data_dir,'test'), data_transforms['test'])
    if (task == '4-Long-Tailed'):
        img_num_per_cls = [0] * 10
        for inputs, labels in image_dataset_train:
            img_num_per_cls[labels] += 1
        if not os.path.exists('cls_freq'):
            os.makedirs('cls_freq')
        freq_path = os.path.join('cls_freq', task + '.json')
        with open(freq_path, 'w') as fd:
            json.dump(img_num_per_cls, fd)
        


    train_loader = DataLoader(image_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(image_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader


def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
    gamma = 1. / imb_factor
    img_max = len(self.data) / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (gamma ** (cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * gamma))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)

    # save the class frequency
    if not os.path.exists('cls_freq'):
        os.makedirs('cls_freq')
    freq_path = os.path.join('cls_freq', self.dataset_name + '_IMBA{}.json'.format(imb_factor))
    with open(freq_path, 'w') as fd:
        json.dump(img_num_per_cls, fd)

    return img_num_per_cls

