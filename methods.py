
import torch
from torchvision import datasets, transforms, models
from torchvision.models import vgg16, VGG16_Weights, densenet121, DenseNet121_Weights
from torchvision.models import VGG16_Weights, VGG13_Weights, DenseNet121_Weights

from torch import nn, optim
import torch.nn.functional as F

model_functions = {
    'vgg16': models.vgg16,
    'vgg13': models.vgg13,
    'densenet121': models.densenet121
}

weights_mapping = {
    'vgg16': VGG16_Weights.DEFAULT,
    'vgg13': VGG13_Weights.DEFAULT,
    'densenet121': DenseNet121_Weights.DEFAULT
}

def load_data(data_directory):
    # Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # Load the datasets with ImageFolder
    image_datasets = {
        x: datasets.ImageFolder(data_directory + '/' + x, transform=data_transforms[x])
        for x in ['train', 'valid']
    }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
        for x in ['train', 'valid']
    }

    return dataloaders

def get_pretrained_model(arch='vgg16', hidden_units=512, class_to_idx=None):
    # Ensure the architecture is available, else default to vgg16
    if arch not in weights_mapping:
        print(f"Architecture '{arch}' not recognized. Defaulting to vgg16.")
        arch = 'vgg16'

    # Load a pre-trained model
    model = models.__dict__[arch](weights=weights_mapping[arch])

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    if arch.startswith('vgg'):
        num_features = model.classifier[0].in_features  # Typically 25088 for VGG models
    elif arch == 'densenet121':
        num_features = model.classifier.in_features    # Typically 1024 for DenseNet121
    else:
        num_features = 25088  # Default case, adjust as necessary

    # Here we use hidden_units in the classifier
    classifier = nn.Sequential(
        nn.Linear(num_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, len(class_to_idx)),  # Output layer size = number of classes
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier  # Adjust this line as necessary

    # Assign class_to_idx as an attribute of the model
    model.class_to_idx = class_to_idx

    return model


def initialize_model(arch='vgg16', hidden_units=512, learning_rate=0.001, class_to_idx=None):
    model = get_pretrained_model(arch, hidden_units, class_to_idx)

    # Define the loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return model, criterion, optimizer

def train_model(model, criterion, optimizer, dataloaders, epochs=5, gpu=False):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0
        print("epoch: ")
        print(epoch)
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation pass
        model.eval()  # Set model to evaluate mode
        validation_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model(inputs)
                batch_loss = criterion(logps, labels)
                validation_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(dataloaders['train']):.3f}.. "
              f"Validation loss: {validation_loss/len(dataloaders['valid']):.3f}.. "
              f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")

        model.train()

def save_checkpoint(model, save_dir, arch='vgg16', hidden_units=512):
    checkpoint = {'arch': arch,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'hidden_units': hidden_units}

    torch.save(checkpoint, f'{save_dir}/checkpoint.pth')
