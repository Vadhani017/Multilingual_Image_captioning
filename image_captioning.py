import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast
from PIL import Image
from translate import Translator

# Load the fine-tuned image captioning model and corresponding tokenizer and image processor
model = VisionEncoderDecoderModel.from_pretrained("Abdou/vit-swin-base-224-gpt2-image-captioning")
tokenizer = GPT2TokenizerFast.from_pretrained("Abdou/vit-swin-base-224-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("Abdou/vit-swin-base-224-gpt2-image-captioning", image_channel_format="RGB")

# Function to perform inference
def get_caption(model, image_processor, tokenizer, image_path):
    image = Image.open(image_path).convert("RGB")
    # Preprocess the image
    img = image_processor(image, return_tensors="pt")
    # Generate the caption (using greedy decoding by default)
    output = model.generate(**img, max_new_tokens=100)  # Adjust max_new_tokens as needed
    # Decode the output
    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return caption

# Streamlit frontend
st.title("Multilingual Image Caption Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

languages = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "te": "Telugu",
    "mr": "Marathi",
    "ta": "Tamil",
    "ur": "Urdu",
    "gu": "Gujarati",
    "kn": "Kannada",
    "pa": "Punjabi"
}

selected_language_name = st.selectbox("Select Language", list(languages.values()))

if uploaded_file is not None:
    st.write("Uploaded Image:")
    temp_image_path = "temp_image.png"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Display uploaded image
    image = Image.open(temp_image_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    caption = get_caption(model, image_processor, tokenizer, temp_image_path)
    
    # Translate caption using 'translate' library
    if selected_language_name != "English":
        selected_language_code = [code for code, name in languages.items() if name == selected_language_name][0]
        translator = Translator(to_lang=selected_language_code)
        translated = translator.translate(caption)
        caption = translated
        
    st.write(f"Generated Caption ({selected_language_name}): {caption}")
