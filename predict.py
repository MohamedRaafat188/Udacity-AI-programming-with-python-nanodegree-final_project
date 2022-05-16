import argparse
import torch
from torchvision import models
from PIL import Image
import numpy as np
import json


def get_args():
    parser = argparse.ArgumentParser(description="Image to classify and the model used in classification")

    parser.add_argument('--image_path', type=str, help='enter the path of the image you want to classify', default='./flower.jpg')
    parser.add_argument('--checkpoint_path', type=str, help='enter the path of the checkpoint you will use in prediction', default='./model_resnet152.pth')
    parser.add_argument('--category_name', type=str, help='enter the path the file contain mapping from categories to classess names', default='./cat_to_name.json')
    parser.add_argument('--topk', type=int, help='number of most predicted classess', default=1)
    parser.add_argument('--gpu', type=bool, help='Use gpu in calculations', default=False)

    args = parser.parse_args()
    return args


def check_gpu(gpu_arg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    return device


def load_mapping(file_path):
    with open(file_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def process_image(image_path):
    image = Image.open(image_path)

    cols, rows = image.size
    if cols >= rows:
        image.thumbnail((cols, 256))
    else:
        image.thumbnail((256, rows))

    cols, rows = image.size
    new_square_size = 224
    left = (cols - new_square_size)/2
    top = (rows - new_square_size)/2
    right = (cols + new_square_size)/2
    bottom = (rows + new_square_size)/2

    image = image.crop((left, top, right, bottom))

    np_image = np.array(image)
    np_image = (np_image/255-[0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]

    return torch.from_numpy(np_image.transpose(2, 0, 1))


def predict(image_path, model, device, topks, cat_to_name_dict):
    image = process_image(image_path)
    image = image.view(1, 3, 224, 224).float()
    
    model.eval()
    
    model, image = model.to(device), image.to(device)
    
    with torch.no_grad():
        output = model.forward(image)
        ps = torch.exp(output)
        top_ps, top_classes = ps.topk(topks, dim=1)
    
    top_ps, top_classes = top_ps.to('cpu'), top_classes.to('cpu')

    top_classes_cat = [list(model.class_to_idx.keys())[c] for c in top_classes.numpy()[0]]
    top_classes_cat = [cat_to_name_dict[c] for c in top_classes_cat]

    return top_ps[0], top_classes_cat


def load_checkpoint(model_path):
    checkpoint = torch.load(model_path)

    if "resnet" in model_path:
        model = models.resnet152(pretrained=True)
        model.fc = checkpoint["classifier"]
    else:
        model = models.densenet121(pretrained=True)
        model.classifier = checkpoint["classifier"]
    
    model.load_state_dict(checkpoint["model state dict"])
    model.class_to_idx = checkpoint['class to index']
    
    return model


def main():
    args = get_args()
    device = check_gpu(args.gpu)
    cat_to_name = load_mapping(args.category_name)
    model = load_checkpoint(args.checkpoint_path)
    top_ps, top_classes = predict(args.image_path, model, device, args.topk, cat_to_name)
    for pair in zip(top_ps, top_classes):
        print(pair)
