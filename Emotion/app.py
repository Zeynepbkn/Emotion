import gradio as gr
import numpy as np
from PIL import Image
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import requests
from io import BytesIO

# Emotion classes
emotions = []
if os.path.exists('model_classes.txt'):
    with open('model_classes.txt', 'r') as f:
        emotions = [line.strip() for line in f.readlines()]
else:
    emotions = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

# Classes compatible with FastAI model
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=None):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
    
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.view(x.size(0), -1)

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

# Complete ResNet34 + FastAI customization
class EmotionResnet34(nn.Module):
    def __init__(self, num_classes=5):
        super(EmotionResnet34, self).__init__()
        
        # First layer - ResNet34's first layer
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Layer1 - 3 BasicBlock
        self.layer1 = self._make_layer(64, 64, 3)
        
        # Layer2 - 4 BasicBlock
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        
        # Layer3 - 6 BasicBlock
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        
        # Layer4 - 3 BasicBlock
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        # FastAI head part
        self.head = nn.Sequential(
            AdaptiveConcatPool2d(),
            Flatten(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes, bias=False)
        )
        
    def _make_layer(self, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
            
        layers = []
        layers.append(BasicBlock(inplanes, planes, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        return x

# Model file path
MODEL_PATH = 'emotion_resnet34.safetensors'

# Create model
model = EmotionResnet34(len(emotions))

# Load state dict
try:
    print("Loading model...")
    if os.path.exists(MODEL_PATH):
        try:
            from safetensors.torch import load_file
            print(f"Loading model with SafeTensors: {MODEL_PATH}")
            state_dict = load_file(MODEL_PATH)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Switching to standard ResNet34 model...")
            # Fallback to standard model
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, len(emotions))
            model.eval()
    else:
        print("Model file not found!")
        print("Switching to standard ResNet34 model...")
        # Fallback to standard model
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, len(emotions))
        model.eval()
except Exception as e:
    print(f"Critical error loading model: {e}")
    # Fallback to standard model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(emotions))
    model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Preprocess the image
def preprocess_image(image):
    if image is None:
        return None
    
    # Convert numpy array to PIL image
    img = Image.fromarray(image).convert('RGB')
    # Apply transformations
    tensor = transform(img)
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor

# Function to make predictions
def predict_emotion(image):
    if image is None:
        return "Please upload an image", None
    
    try:
        # Preprocess the image
        tensor = preprocess_image(image)
        
        # Use GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        tensor = tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)[0]
        
        # Select the emotion with highest probability
        _, predicted = torch.max(outputs, 1)
        emotion = emotions[predicted.item()]
        
        # Probabilities for all emotions
        confidence = {emotions[i]: float(probs[i].cpu()) for i in range(len(emotions))}
        
        return emotion, confidence
    except Exception as e:
        return f"Error occurred: {str(e)}", None

# Function to fetch remote image
def fetch_image(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return np.array(img)
    except:
        return None

# Gradio interface
with gr.Blocks(title="Emotion Recognition") as demo:
    gr.Markdown("# Emotion Recognition Application")
    gr.Markdown("This application recognizes facial expressions (Angry, Happy, Neutral, Sad, Surprise) from images.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Image", type="numpy")
            submit_btn = gr.Button("Predict")
        
        with gr.Column(scale=1):
            output_emotion = gr.Textbox(label="Predicted Emotion")
            output_confidence = gr.Label(label="Confidence Levels")
    
    submit_btn.click(
        fn=predict_emotion, 
        inputs=input_image,
        outputs=[output_emotion, output_confidence]
    )
    
    # Example images section
    gr.Markdown("### Example Images")
    gr.Markdown("Click on any example to analyze:")
    
    # Create examples
    example_pairs = []
    example_descriptions = [
        "Angry example face",
        "Happy example face",
        "Neutral example face", 
        "Sad example face",
        "Surprise example face"
    ]
    
    # Add examples with Gradio Examples component
    gr.Examples(
        examples=[
            ["sample/sample1.jpg", "Angry example face"],
            ["sample/sample2.jpg", "Happy example face"],
            ["sample/sample3.jpg", "Neutral example face"],
            ["sample/sample4.jpg", "Sad example face"],
            ["sample/sample5.jpg", "Surprise example face"]
        ],
        inputs=input_image,
        outputs=[output_emotion, output_confidence],
        fn=predict_emotion,
        examples_per_page=5,
        label="Emotion Examples"
    )

# Start the application
if __name__ == "__main__":
    print("Starting application...")
    # Settings suitable for Hugging Face Spaces
    demo.launch(share=False, show_error=True) 
