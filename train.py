import argparse
import torch
from torchvision import transforms, models, datasets
from torch import nn
from torch import optim
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser(description="Neural Network characterstics.")

    parser.add_argument('--data_dir', type=str, help='enter the directory of the data.', default='flowers/')
    parser.add_argument('--save_dir', type=str, help='enter the directory to save the trained model in.', default='./')
    parser.add_argument('--model_arch', type=str, help='enter the architecture of the model', default='resnet152')
    parser.add_argument('--learning_rate', type=float, help='enter the learning rate (the perfect learning rate is a float number between 0.001 and 0.01)', default=0.0015)
    parser.add_argument('--hidden_units', type=int, help='number of hidden units in hidden layers', default=512)
    parser.add_argument('--epochs ', type=int, help='number of times the model will loop on the data to learn and enhance its prediction', default=10)
    parser.add_argument('--gpu', type=bool, help='Use gpu in calculations', default=False)

    args = parser.parse_args()
    return args


def load_process_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(30),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                      [0.229, 0.224, 0.225])]),
                                                            
    'valid': transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                      [0.229, 0.224, 0.225])]),
    'test': transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                     [0.229, 0.224, 0.225])])
    }

    image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
    'valid': DataLoader(image_datasets['valid'], batch_size=64, shuffle=True),
    'test': DataLoader(image_datasets['test'], batch_size=64, shuffle=True)
    }

    return dataloaders


def check_gpu(gpu_arg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    return device


def build_model(model_name, hidden_units, learning_rate):
    if 'resnet' in model_name:
        model = models.resnet152(pretrained=True)
        model.name = "resnet152"
        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Sequential(nn.Linear(2048, hidden_units), 
                                 nn.ReLU(), 
                                 nn.BatchNorm1d(hidden_units),
                                 nn.Dropout(0.4),
                                 nn.Linear(hidden_units, 102),
                                 nn.LogSoftmax(dim=1))
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    elif 'densenet' in model_name:
        model = models.densenet121(pretrained=True)
        model.name = "densenet121"
        for param in model.parameters():
            param.requires_grad = False

        model.classifier = nn.Sequential(nn.Linear(1024, hidden_units), 
                                         nn.ReLU(), 
                                         nn.BatchNorm1d(hidden_units),
                                         nn.Dropout(0.5),
                                         nn.Linear(hidden_units, 102),
                                         nn.LogSoftmax(dim=1))
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    else:
        print("Sorry you can only use densenet121 ro resnet152 architectures")
        model = None
        optimizer = None

    return model, optimizer
    

def train_evaluate_model(model, train_data, valid_data, optimizer, epochs, device):
    criterion = nn.NLLLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

    model.to(device)

    running_loss = 0

    for epoch in range(epochs):
        for images, labels in train_data:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            model.eval()
            valid_loss = 0
            accuracy = 0
            with torch.no_grad():
                for images, labels in valid_data:
                    images, labels = images.to(device), labels.to(device)
                    logps = model.forward(images)

                    batch_loss = criterion(logps, labels)
                    valid_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            scheduler.step(valid_loss)

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/len(train_data):.3f}.. "
                  f"Test loss: {valid_loss/len(valid_data):.3f}.. "
                  f"Test accuracy: {accuracy/len(valid_data):.3f}")

            running_loss = 0
            model.train()
    return model


def test_model(model, test_data, criterion, device):
    model.to(device)
    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for images, labels in test_data:
            images, labels = images.to('cuda'), labels.to('cuda')
            logps = model.forward(images)

            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(test_loss / len(test_data), accuracy / len(test_data))


def save_checkpoint(model, optimizer, save_dir, train_data, epochs):
    checkpoint = {
        "classifier": model.fc,
        "class to index": train_data.class_to_idx,
        "model state dict": model.state_dict(),
        "epochs": epochs,
        "optimizer": optimizer,
        "optimizer_state_dict": optimizer.state_dict()
    }
    if save_dir[-1] == '/':
        torch.save(checkpoint, save_dir + model.name)
    else:
        torch.save(checkpoint, save_dir + '/' + model.name)


def main():
    args = get_args()
    data = load_process_data(args.data_dir)
    device = check_gpu((args.gpu))
    model, optimizer = build_model(args.arch, args.hidden_units, args.learning_rate)
    print("Training Process")
    model = train_evaluate_model(model, data['train'], data['valid'], optimizer, args.epochs, device)
    print("Testing the model on the test data")
    test_model(model, data['test'], nn.NLLLoss(), device)
    save_checkpoint(model, optimizer, args.save_dir, data['train'], args.epochs)


if __name__ == "__main__":
    main()
