import torch
import torchvision.models as models

print("Loading ResNet50...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

print("Tracing model...")
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

print("Saving model...")
torch.jit.save(traced_model, "model_repository/resnet50/1/model.pt")
print("Done! Model saved to model_repository/resnet50/1/model.pt")
