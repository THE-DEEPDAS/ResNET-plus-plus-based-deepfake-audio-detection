import torch
from torchviz import make_dot
from finale import ResNetPlusPlus

def visualize_model_architecture():
    model = ResNetPlusPlus(num_classes=2)
    x = torch.randn(1, 3, 224, 224)  # Dummy input for visualization
    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render('model_architecture')

if __name__ == "__main__":
    visualize_model_architecture()
