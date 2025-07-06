
from flask import Flask, request, render_template
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Combined Model class
class CombinedModel(nn.Module):
    def __init__(self, model_effnet, model_mobilenet):
        super().__init__()
        self.model_effnet = model_effnet
        self.model_mobilenet = model_mobilenet

    def forward(self, x):
        prob_effnet = torch.softmax(self.model_effnet(x), dim=1)
        prob_mobilenet = torch.softmax(self.model_mobilenet(x), dim=1)
        avg_prob = (prob_effnet + prob_mobilenet) / 2
        return avg_prob

# Initialize models
model_effnet = models.efficientnet_b0(pretrained=False)
model_effnet.classifier = nn.Sequential(
    nn.Flatten(), nn.Linear(1280, 1024), nn.ReLU(),
    nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 3)
)
model_mobilenet = models.mobilenet_v2(pretrained=False)
model_mobilenet.classifier[1] = nn.Linear(model_mobilenet.last_channel, 3)

# Create combined model
combined_model = CombinedModel(model_effnet, model_mobilenet).to(device)
combined_model.load_state_dict(torch.load('combined_model.pth', map_location=device))
combined_model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    if request.method == 'POST':
        file = request.files['file']
        img = Image.open(file).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = combined_model(img)
            _, predicted = torch.max(outputs, 1)
        result = f'Predicted class: {predicted.item()}'
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
