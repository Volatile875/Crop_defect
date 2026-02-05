import requests
from PIL import Image
import io
import time
import sys

def create_dummy_image():
    # Create a 224x224 RGB image (greenish)
    img = Image.new('RGB', (224, 224), color = (73, 109, 137))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def test_predict_image():
    url = "http://localhost:8000/predict_image"
    try:
        image_data = create_dummy_image()
        files = {'file': ('test.jpg', image_data, 'image/jpeg')}
        
        print(f"Sending request to {url}...")
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            print("✅ Success! Response:")
            print(response.json())
        else:
            print(f"❌ Failed with status code {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Wait for server to start
    time.sleep(5) 
    test_predict_image()
