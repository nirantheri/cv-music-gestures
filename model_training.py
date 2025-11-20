import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, transforms
import time
import os
import copy
import matplotlib.pyplot as plt
from simple_model import Classifier, SimpleClassifier
from tqdm import tqdm
import pandas as pd
# %matplotlib inline
plt.rcParams['figure.figsize'] = [16, 10]

#******Helper functions********
# These are all as seen in class (with slight modifications to adapt them)
# and do not need to be modified by you.

def tensor_show(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

def visualize_model(model, dataloaders, class_names, device, num_images=6):
    """Visualize model predictions."""
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                tensor_show(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

#******Dataloader functions********
# You are provided with a dataloader that takes in a version argument.
# You will be asked questions about the affect of the version argument,
# but don't need to modify this code.

def build_dataloader(data_dir, version = 0):
    """Construct a dataloader.
    Input:
    - data_dir: The root folder of the dataset. Data is assumed to be organized
      as shown in the transfer learning exercise in class.
    - version: A version flag (value 0 or 1) that swaps between two different
      transform specifications.
    Returns:
    - dataloaders: A torch dataloader object that provides data to the training
      procedure.
    - datadict: A formatted dictionary with useful information about the data.
    """
    if version == 0:
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    elif version == 1:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    else:
        error("Not a recognized dataloader version flag.")

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                          data_transforms[x])
                     for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
                     for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    datadict = {
        "class_names": class_names,
        "num_classes": len(class_names),
        "dataset_sizes": dataset_sizes,
    }

    return dataloaders, datadict

#******Model Handling********
# Here you are provided with part of a model setup function, and you will need
# to provide the rest of the implementation.

#TODO: Finish this function by adding lines at the TODO marked below.
def setup_model(datadict, device):
    """Set up a model for transfer learning according to the description
    given in the assignment notebook.
    Input:
    - datadict: A formatted dictionary with useful information about the
      data. The fields are defined in build_dataloader() above.
    - device: The torch device where we will be performing computations.
    Returns:
    - model_dict: A formatted dictionary of model information consisting of:
        --model: A network with layers set up ready to be trained for
          recognition on our custom dataset.
        --criterion: The loss function.
        --optimizer: The optimizer.
        --scheduler: The learning rate scheduler.
    """

    # we are going to be using VGG16 for our transfer learning.
    model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    # we have preset a block of the fully-connected layers to be pass-through
    # instead.
    for l in [3,4,5]:
        model.classifier[l] = nn.Identity()

  
    # number of final outputs = number of classes
    out_feat = datadict["num_classes"]

    model.classifier[len(model.classifier)-1] = nn.Linear(in_features=4096, out_features=out_feat)

    # this command sends the model to our device
    model = model.to(device)
    
    # here we set what type of loss we plan to use. Since this is a recognition task,
    # cross entropy is a good loss function.
    criterion = nn.CrossEntropyLoss()

    # we also need to set up an optimizer. Our standard SGD works fine.
    optimizer = optim.SGD(model.classifier[3:].parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    if torch.cuda.is_available():
        model.to('cuda') # or model.cuda()
    model_dict = {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler
    }

    return model_dict

def setup_simple_model(datadict, class_n, device):
    # initialize model
    # get input size and number of classes
    num_classes = datadict["num_classes"]

    # for now, 3 is the number of channels
    network_init = {"1": Classifier(3, num_classes), "2": SimpleClassifier(150528, num_classes)}

    model = network_init[class_n]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    if torch.cuda.is_available():
        model.to('cuda') # or model.cuda()
    model_dict = {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler
    }

    return model_dict

#******Training Setup********
# Here you are provided with part of training procedure, and you will need to
# complete the code.

def train_model(model_dict, dataloaders, datadict, device, num_epochs=15):
    """
    Defines a training procedure for our transfer learning problem.
    Inputs:
    - model_dict: Our model to be trained and its accompanying information.
    - dataloaders: Our training and validation dataloader.
    - datadict: A data dictionary with information about the data being
      processed; this will be useful for data logging.
    - device: The torch device performing the evaluations.
    - num_epochs: The number of loops through our training data.
    Outputs:
    - model: The version of our trained model with the best validation score.
    - logs: A dictionary of performance logs.
    """

    since = time.time()

    # We are going to run for a set number of epochs, but that doesn't mean our final epoch is our best.
    # Keep track of which version of the model worked the best.
    best_model_wts = copy.deepcopy((model_dict["model"]).state_dict())
    best_acc = 0.0

    numval = datadict["dataset_sizes"]["val"]/datadict["num_classes"]

    # data logging
    losslog = [[],[]]
    acclog = [[],[]]
    classlog = [] # this is the list you will use to append class accuracy
    num_classes = datadict['num_classes']
    class_names = datadict['class_names']

    for epoch in tqdm(range(num_epochs)):

        class_correct = np.zeros(num_classes)
        class_total = np.zeros(num_classes)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                (model_dict["model"]).train()  # Set model to training mode
            else:
                (model_dict["model"]).eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                model_dict["optimizer"].zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_dict["model"](inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = model_dict["criterion"](outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        (model_dict["optimizer"]).step()
                    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if phase=='val':
                    for i in range(len(labels)):
                        label = labels[i].item() # item makes it a number
                        pred = preds[i].item()
                        class_total[label] += 1   
                        if label == pred:
                            class_correct[label] += 1
        
            if phase == 'train':
                model_dict["scheduler"].step()

            epoch_loss = running_loss / datadict["dataset_sizes"][phase]
            epoch_acc = running_corrects.double() / datadict["dataset_sizes"][phase]

            # data logging
            if phase == 'train':
                losslog[0].append(epoch_loss)
                acclog[0].append(float(epoch_acc))
            else:
                losslog[1].append(epoch_loss)
                acclog[1].append(float(epoch_acc))           


            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model_dict["model"].state_dict())
            # integrate the classlog
        
        # Compute per-class accuracy safely
        raw_acc = np.divide(
            class_correct,
            class_total,
            out=np.zeros_like(class_correct),
            where=class_total != 0
        )

        # Convert to dict using class names
        class_acc = {
            class_names[i]: raw_acc[i]
            for i in range(num_classes)
        }

        classlog.append(class_acc)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    (model_dict["model"]).load_state_dict(best_model_wts)

    logs = {
        "train_loss": losslog[0],
        "val_loss": losslog[1],
        "train_acc": acclog[0],
        "val_acc": acclog[1],
        "class_acc": classlog
    }

    return model_dict, logs

if __name__ == "__main__":
    epochs = 50
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = '/cs/cs153/datasets/music_gestures' # this line assumes you are working from the CS servers.

    dataloader_v1, datadict = build_dataloader(data_dir, version = 1)
    model1_dict = setup_simple_model(datadict, "1", device)
    model2_dict = setup_simple_model(datadict, "2", device)
    model3_dict = setup_model(datadict, device)

    model1_dict, log1 = train_model(model1_dict, dataloader_v1, datadict, device, num_epochs=epochs)
    model2_dict, log2 = train_model(model2_dict, dataloader_v1, datadict, device, num_epochs=epochs)
    model3_dict, log3 = train_model(model3_dict, dataloader_v1, datadict, device, num_epochs=epochs)

    model_1 = model1_dict["model"]
    model_2 = model2_dict["model"]
    model_3 = model3_dict["model"]

    path1 = "model_state_dict_1.pth"
    path2 = "model_state_dict_2.pth"
    path3 = "model_state_dict_3.pth"
    torch.save(model_1.state_dict(), path1)
    torch.save(model_2.state_dict(), path2)
    torch.save(model_3.state_dict(), path3)

    if not os.path.exists('output/'):
        os.mkdir('output')

    fig = plt.figure()

    ax = fig.add_subplot(1,3,1)
    ax.set_ylim([0.6,1.0])
    plt.plot(log1["train_acc"], 'r-')
    plt.plot(log1["val_acc"], 'b-')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for Model V1')
    plt.legend(['training', 'validation'])

    ax = fig.add_subplot(1,3,2)
    ax.set_ylim([0.6,1.0])
    plt.plot(log2["train_acc"], 'r-')
    plt.plot(log2["val_acc"], 'b-')
    plt.title('Accuracy for Model V2')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.legend(['training', 'validation'])

    ax = fig.add_subplot(1,3,3)
    ax.set_ylim([0.6,1.0])
    plt.plot(log3["train_acc"], 'r-')
    plt.plot(log3["val_acc"], 'b-')
    plt.title('Accuracy for Model V3')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.legend(['training', 'validation'])

    fig.savefig('output/model_acc.png', bbox_inches='tight', pad_inches=0)

    df_1=pd.DataFrame(log1)
    df_2=pd.DataFrame(log2)
    df_3=pd.DataFrame(log3)

    df_1.to_csv('output_4.csv', index=False)
    df_2.to_csv('output_5.csv', index=False)
    df_3.to_csv('output_6.csv', index=False)