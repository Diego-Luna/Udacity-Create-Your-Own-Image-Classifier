import argparse
import torch
from PIL import Image
import json
import numpy as np
import methods  # make sure this points to your methods file

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = methods.get_pretrained_model(arch=checkpoint['arch'], class_to_idx=checkpoint['class_to_idx'])
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Open the image
    img = Image.open(image_path)

    # Resize the image where the shortest side is 256 pixels, keeping the aspect ratio
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))

    # Crop out the center 224x224 portion of the image
    left_margin = (img.width-224)/2
    top_margin = (img.height-224)/2
    img = img.crop((left_margin, top_margin, left_margin+224, top_margin+224))

    # Convert image to numpy array
    np_image = np.array(img)/255

    # Normalize each color channel
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stds

    # Set the color channel to be the first dimension
    np_image = np_image.transpose((2, 0, 1))

    return np_image

def predict(image_path, model, topk=5, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()

    # Process the image
    processed_img = process_image(image_path)
    processed_img = torch.from_numpy(processed_img).type(torch.FloatTensor)
    processed_img.unsqueeze_(0)  # Add batch dimension

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    processed_img = processed_img.to(device)

    with torch.no_grad():
        output = model.forward(processed_img)

    probs, indices = torch.exp(output).topk(topk)
    probs = probs.cpu().numpy().flatten()
    indices = indices.cpu().numpy().flatten()

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices]

    return probs, classes

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Predict flower name from an image along with the probability")
    parser.add_argument('input', type=str, help='Path to image')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--top_k', type=int, help='Return top KK most likely classes', default=5)
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    # Load model from checkpoint
    model = load_checkpoint(args.checkpoint)

    # Make prediction
    probs, classes = predict(args.input, model, args.top_k, args.gpu)

    # Convert classes to names
    names = []
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        names = [cat_to_name[str(cls)] for cls in classes]
    else:
        names = classes

    # Print out probabilities and classes
    print("Probabilities:", probs)
    print("Classes:", names)

if __name__ == '__main__':
    main()
