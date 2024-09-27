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

def get_middle_frame(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        raise IOError("Error opening video file.")

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the index of the middle frame
    middle_frame_index = (total_frames - 1) // 2

    # Set the position of the next read
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)

    # Read the middle frame
    ret, frame = cap.read()

    # Release the video capture object
    cap.release()

    # Check if frame reading was successful
    if not ret:
        raise ValueError("Error reading the middle frame.")

    return frame
