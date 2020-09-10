import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
from PIL import Image


def load_the_data(flower_dataset = './flowers'):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(225),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    valid_transform = transforms.Compose([transforms.Resize(225),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transform)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    
    return train_data, trainloader, testloader, validloader

def label_mapping():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def define_network(arch='vgg19', hidden_layer =2048, dropout=0.5, lr =0.001, device='gpu'):
    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_units = 25088
        hidden_units = hidden_layer
        output_units = len(label_mapping())
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_units = 25088
        hidden_units = hidden_layer
        output_units = len(label_mapping())
    else:
        print("\n\nAvailable models are VGG16 and VGG19 and the default model is VGG19!\n")
        model = models.densenet169(pretrained=True)
        input_units = 25088
        hidden_units = hidden_layer
        output_units = len(label_mapping())
        
    for parameter in model.parameters():
        parameter.requires_grad = False
        
    model.classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088,2048)),
    ('relu1', nn.ReLU()),
    ('dropout1', nn.Dropout(p=0.5)),
    ('fc2', nn.Linear(2048, 102)),
#     ('relu2', nn.ReLU()),
#     ('dropout2', nn.Dropout(p=0.5)),
#     ('fc3', nn.Linear(512, 102)),
    ('log_softmax', nn.LogSoftmax(dim=1))
    ]))             
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr) 
    
    return model, criterion, optimizer

def train_network(model, criterion, optimizer, trainloader, validloader, epoch, gpu):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epoch
    running_loss = 0
    steps = 0
    print_every = 5
    print("\n--------------I am getting trained, please Wait!------------- \n")

    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        log_ps = model(images)
                        valid_loss += criterion(log_ps, labels)
                        ps = torch.exp(log_ps)
                        top_prob, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
                
    print("\n-------------- Your model has been trained Successfully -----------------------\m")
    


def save_checkpoint(file_path, model, optimizer, train_data, args):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'arch': args.arch, 
                  'model': model,
                  'epochs': args.epochs,
                  'learning_rate': args.learning_rate,
                  'hidden_units': args.hidden_units,
                  'classifier' : model.classifier,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, file_path) 
        
def load_checkpoint(file_path):
    checkpoint = torch.load(file_path)
    epochs = checkpoint['epochs']
    learning_rate = checkpoint['learning_rate']
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    transform_image = transforms.Compose([transforms.Resize(225),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    PIL_img = transform_image(img)
    
    return PIL_img

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    model.cpu()
    image = process_image(image_path)
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        log_ps = model(image)
        ps = torch.exp(log_ps)
        top_prob, top_class = ps.topk(topk, dim=1)

        idx_to_class = {value: key for key, value in model.class_to_idx.items()}

        classes = []

        for index in top_class.numpy()[0]:
            classes.append(idx_to_class[index])

        return top_prob.numpy()[0], classes
