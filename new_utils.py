import cloudinary
import cloudinary.uploader

# Configure Cloudinary with your credentials
cloudinary.config(
    cloud_name="ASSETS_images",
    api_key="317327128284384",
    api_secret="lD_fcfCEbwfFTnhoDjomJUDgt5c"
)

def upload_and_delete_image(image_path):
    # Upload image
    upload_result = cloudinary.uploader.upload(image_path)
    image_url = upload_result['secure_url']
    
    # You now have the URL, do whatever you need with it.
    print(f"Image URL: {image_url}")
    
    # After a few minutes, you can delete the image using its public ID
    public_id = upload_result['public_id']
     # Wait before deleting

    return image_url, public_id
    
    # Delete the image
cloudinary.uploader.destroy(public_id)
print("Image deleted successfully.")
