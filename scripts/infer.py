import tritonclient.http as httpclient
import numpy as np
import sys

TRITON_IP = sys.argv[1] if len(sys.argv) > 1 else "localhost"

print(f"Connecting to Triton at {TRITON_IP}:8000...")
client = httpclient.InferenceServerClient(url=f"{TRITON_IP}:8000")

# Check server is ready
print(f"Server ready: {client.is_server_ready()}")
print(f"Model ready: {client.is_model_ready('resnet50')}")

# Create random input (simulating an image)
image_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

# Build request
inputs = [httpclient.InferInput("input__0", image_data.shape, "FP32")]
inputs[0].set_data_from_numpy(image_data)
outputs = [httpclient.InferRequestedOutput("output__0")]

# Send inference
print("Sending inference request...")
response = client.infer("resnet50", inputs, outputs=outputs)
result = response.as_numpy("output__0")

print(f"Output shape: {result.shape}")
print(f"Top predicted class index: {np.argmax(result[0])}")
print(f"Top 5 class indices: {np.argsort(result[0])[-5:][::-1]}")
print("SUCCESS - Triton inference working!")
