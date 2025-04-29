from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils import get_dataset, ARGS
from update import test_inference
from models import CNN_CBISDDSM
import os
from PIL import Image
from datasets import CBISDDSM_Dataset

# Paths
csv_folder = r'C:\Users\chari\OneDrive\Desktop\422128\MiniProject-II\archive\csv'
jpeg_folder = r'C:\Users\chari\OneDrive\Desktop\422128\MiniProject-II\archive\jpeg'

# 1. Load CSV file
csv_files = [file for file in os.listdir(csv_folder) if file.endswith('.csv')]

if len(csv_files) == 0:
    raise FileNotFoundError("No CSV files found in the folder.")

csv_file_path = os.path.join(csv_folder, csv_files[0])
df = pd.read_csv(csv_file_path)

#print(f"Loaded CSV: {csv_file_path}")
#print(df.head())

# 2. Check the columns in CSV and identify the relevant image column
#print("CSV columns:", df.columns)

image_column = 'cropped image file path'

# 3. Function to load image from CSV path
import pydicom
import matplotlib.pyplot as plt


def load_image(image_id):
    parts = image_id.split('/')
    uid = parts[-1]
    img_folder = os.path.join("C:/Users/chari/OneDrive/Desktop/422128/MiniProject-II/archive/jpeg", uid)
    img_file = os.path.join(img_folder, "000001.dcm")

    print("Trying to load:", img_file)

    if not os.path.exists(img_file):
        raise FileNotFoundError(f"Image not found: {img_file}")

    # Read DICOM image using pydicom
    dicom_image = pydicom.dcmread(img_file)
    img_array = dicom_image.pixel_array
    img = Image.fromarray(img_array)  # Convert to PIL Image for further processing

    return img

# 4. Main execution
if __name__ == '__main__':
    args = ARGS()
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # Load datasets using get_dataset (Ensure this is implemented in utils.py)
    train_dataset, test_dataset, _ = get_dataset(args)
    print("Model selected:", args.model)
    print("Dataset selected:", args.dataset)

    # BUILD MODEL
    if args.model == 'cnn':
        # Use CBISDDSM_CNN model for CBIS-DDSM dataset
        if args.dataset == 'cbis_ddsm':
            global_model = CNN_CBISDDSM(args=args)
    else:
        exit('Error: unrecognized model')

    # Send the model to the appropriate device (GPU or CPU)
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    # Prepare DataLoader for training
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    epoch_loss = []

    # Training loop
    for epoch in tqdm(range(args.epochs)):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss) / len(batch_loss)
        print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)

    # Plot training loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                 args.epochs))
    plt.title('Training Loss Over Epochs')
    plt.grid(True)

    # Testing the model
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100 * test_acc))

# Log the model and dataset being used
print(f"Running experiment with model: {args.model}")
print(f"Using dataset: {args.dataset}")
