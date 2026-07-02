# from transformers import AutoImageProcessor, AutoModel
# import torch
# from PIL import Image
# import numpy as np

# # Load the model and processor
# model_name = "facebook/dinov2-large"  # or 'facebook/dinov2-base'
# processor = AutoImageProcessor.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')

# def get_dinov2_embedding(image: Image.Image) -> np.ndarray:
#     """Extracts a normalized embedding for a single image using DINOv2."""
#     with torch.no_grad():
#         inputs = processor(images=image, return_tensors="pt").to(model.device)
#         outputs = model(**inputs)
#         # The 'last_hidden_state' contains [CLS] token + patch embeddings
#         cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
#     # L2 normalize
#     return cls_embedding / np.linalg.norm(cls_embedding)


from transformers import AutoModel, AutoImageProcessor

model_name = "facebook/dinov2-large"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)