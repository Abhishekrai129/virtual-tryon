import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

# Define the VirtualTryOnModel class
class VirtualTryOnModel:
    def __init__(self):
        pass

    def apply_virtual_try_on(self, clothing_image, user_image):
        # Ensure both images have 3 channels (RGB) before proceeding
        if clothing_image.mode != 'RGB':
            clothing_image = clothing_image.convert('RGB')
        if user_image.mode != 'RGB':
            user_image = user_image.convert('RGB')

        # Apply transformations (resize and convert to tensor)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # Transform the images
        clothing_image_tensor = transform(clothing_image).unsqueeze(0)  # Add batch dimension
        user_image_tensor = transform(user_image).unsqueeze(0)

        # Ensure both tensors have the same size for all dimensions
        if clothing_image_tensor.size(1) != user_image_tensor.size(1):
            raise RuntimeError(f"Image channel size mismatch: "
                               f"clothing has {clothing_image_tensor.size(1)} channels, "
                               f"user image has {user_image_tensor.size(1)} channels.")

        # Simulate applying the virtual try-on effect (average the images as a dummy effect)
        result_tensor = (clothing_image_tensor + user_image_tensor) / 2.0
        
        # Convert the result back to a PIL image
        result_image = transforms.ToPILImage()(result_tensor.squeeze(0))  # Remove batch dimension
        
        return result_image

# Streamlit app
def main():
    st.title("Virtual Try-On Application")
    st.write("Upload a clothing item and a user photo to try on the clothing virtually.")

    clothing_image_file = st.file_uploader("Upload Clothing Image", type=["jpg", "png"])
    user_image_file = st.file_uploader("Upload User Image", type=["jpg", "png"])

    if clothing_image_file and user_image_file:
        # Open the uploaded files as images
        clothing_image = Image.open(clothing_image_file)
        user_image = Image.open(user_image_file)
        
        # Initialize the model and apply virtual try-on
        model = VirtualTryOnModel()
        result_image = model.apply_virtual_try_on(clothing_image, user_image)

        # Display the result
        st.image(result_image, caption="Virtual Try-On Result", use_column_width=True)

if __name__ == "__main__":
    main()
