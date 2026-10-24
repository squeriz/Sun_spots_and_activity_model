import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import joblib

data = pd.read_csv("dataset/labels.csv")

train_df, test_df = train_test_split(data, test_size=0.2, random_state=12)



class SunSpotDataset(Dataset):
    def __init__ (self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['filepath']
        target_count = float(row['sunspot_number'])

        image = Image.open(f"dataset/{img_path}").convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(target_count, dtype=torch.float32)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class SunspotModelRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.resnet(x).squeeze(-1)

def train_CV(device):
    sunspot_dataset = SunSpotDataset(train_df, transform)

    cv_model = SunspotModelRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(cv_model.parameters(), lr=0.001)

    train_loader = DataLoader(sunspot_dataset, 16, shuffle=True)


    cv_model.train()
    for epoch in range(20):
        total_loss = 0.0
        all_targets = []

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            predictions = cv_model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            all_targets.extend(targets.cpu().numpy())

        target_arr = np.array(all_targets)
        D = np.var(target_arr)

        mse = total_loss / len(sunspot_dataset)
        rmse = np.sqrt(mse)
        r2 = 1 - (mse / D)

        print("MSE: ", mse)
        print("RMSE: ", rmse)
        print("R2: ", r2)
        
        torch.save(cv_model.state_dict(), "sunspot_regressor.pth")


def cv_eval(device, test_df, transform):

    test_dataset = SunSpotDataset(test_df, transform)
    test_loader = DataLoader(test_dataset, 16, shuffle=True)
    cv_model = SunspotModelRegressor()
    cv_model.load_state_dict(torch.load("sunspot_regressor.pth"))
    cv_model.to(device)
    cv_model.eval()

    total_loss = 0.0


    criterion = nn.MSELoss()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            output = cv_model(images)
            loss = criterion(output, targets)
            total_loss += loss.item() * images.size(0)

            all_preds.extend(output.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        target_arr = np.array(all_targets)
        D = np.var(target_arr)
    mse = total_loss / len(test_dataset)
    rmse = np.sqrt(mse)
    r2 = 1 - (mse / D)

    print(f"Test loss (MSE): {mse:.4f}")
    print(f"Average loss (RMSE): {rmse:.4f} spots")
    print(f"R2: {r2:.4f}")

    print("\n Examples matching (y and predict)")
    for i in range(5):
        print(f"True spots: {all_targets[i]:.2f} --> model predict: {all_preds[i]:.2f}")
        


def RF_model():
    data = pd.read_csv("dataset/labels.csv")

    print(data.describe())

    X = pd.DataFrame(data['sunspot_number'])
    y = data['class_4']


    preprocessing_model = LabelEncoder()

    y = preprocessing_model.fit_transform(y)

    print(preprocessing_model.classes_)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)


    model = RandomForestClassifier(n_estimators=1000, max_depth=2, min_samples_leaf=2, max_leaf_nodes=3)

    model.fit(X_train, y_train)

    joblib.dump(model, "random_forest_classifier.pkl")
    joblib.dump(preprocessing_model, "label_encoder.pkl")
    print("RF_model and encoder saved")


    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)


    

    print("-----f1")

    print(f1_score(y_train, pred_train, average="macro"))
    print(f1_score(y_test, pred_test, average="macro"))
    print("--------")

    print()

    print("-------precision/recall")
    print(precision_score(y_train, pred_train, average="macro"))
    print(recall_score(y_test, pred_test, average="macro"))
    print("--------------")
    print()

    pred_prob_train = model.predict_proba(X_train)
    pred_prob_test = model.predict_proba(X_test)

    print("-------ROC-AUC")
    print(roc_auc_score(y_train, pred_prob_train, multi_class="ovr", average="macro"))
    print(roc_auc_score(y_test, pred_prob_test, multi_class="ovr", average="macro"))
    print("-------------------")
    print()


    print(y_test[:5])
    print()

    print(model.predict(X_test[:5]))

# train_CV(device)

# model_test = cv_eval(device, test_df, transform)

# RF_model()


def result_model_prod(image_path, device, transform):
    cv_model = SunspotModelRegressor()
    cv_model.load_state_dict(torch.load("sunspot_regressor.pth"))
    cv_model.to(device)

    cv_model.eval()

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        spots = cv_model(image_tensor).item()

    random_forest_model = joblib.load("random_forest_classifier.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    X_input = pd.DataFrame({'sunspot_number': [spots]})

    pred_encoded = random_forest_model.predict(X_input)[0]

    pred_class = label_encoder.inverse_transform([pred_encoded])[0]

    return spots, pred_class



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_prod_path = "prod/20170104_HMIIC.jpg"

pred_spots, pred_classes = result_model_prod(image_prod_path, device, transform)


print(f"cv_model predicted: {pred_spots:.2f}")
print(f"rf_model predict for this spots count: {pred_classes}")