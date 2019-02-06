# import the usual resources
import torch

from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from util import *

from options import options
options = options()
opts = options.parse()

from models import Net

net = Net(1).to(device)

from torch.utils.data import DataLoader
from torchvision import transforms

from data_load import FacialKeypointsDataset
from data_load import Rescale, RandomCrop, Normalize, ToTensor, RotateScale, HorizontalFlip, VerticalFlip

data_transform = transforms.Compose([RotateScale(60),
                                     Rescale((256, 256)),
                                     RandomCrop((224, 224)),
                                     Normalize(),
                                     ToTensor()])
batch_size = opts.batch

transformed_dataset = FacialKeypointsDataset(csv_file='/data/training_frames_keypoints.csv', root_dir='/data/training/', transform=data_transform)
train_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

test_dataset = FacialKeypointsDataset(csv_file='/data/test_frames_keypoints.csv', root_dir='/data/test/', transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

import torch.optim as optim

criterion = opts.criterion
optimizer = optim.Adam(net.parameters(), lr=opts.lr)


def train_net(n_epochs, model):
    # prepare the net for training
    model.train()
    last100_loss = deque(maxlen=100)

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            key_pts = key_pts.view(key_pts.size(0), -1)

            key_pts = key_pts.type(torch.FloatTensor).to(device)
            images = images.type(torch.FloatTensor).to(device)

            output_pts = model(images)

            loss = criterion(output_pts, key_pts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last100_loss.append(loss.item())
            if batch_i % 100 == 0:  # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i + 1, np.mean(last100_loss)))
                running_loss = 0.0

    print('Finished Training')

train_net(opts.epochs, net)

# get a sample of test data again
test_images, test_outputs, gt_pts = net_sample_output(test_loader, net)

visualize_output(test_images.cpu(), test_outputs.cpu(), gt_pts, batch_size=opts.batch)

model_dir = 'saved_models/'
model_name = 'final_model_augmented_300epoch_no_batchnorm.pt'

torch.save(net.state_dict(), model_dir+model_name)










