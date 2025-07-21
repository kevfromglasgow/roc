import streamlit as st
import tempfile
import os
import subprocess
import zipfile
from pathlib import Path
from PIL import Image
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Spacer, Image as RLImage, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
import io
import shutil
import cv2
import numpy as np
import easyocr
import re

# Set page config
st.set_page_config(
    page_title="ROC Photo Extraction",
    page_icon="🎬",
    layout="wide"
)

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["passwords"]["app_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

def extract_frames(video_path, interval_seconds, output_format, original_filename):
    """Extract frames from video using ffmpeg"""
    
    # Create temporary directory for extracted frames
    temp_dir = tempfile.mkdtemp()
    
    # Use original video filename without extension instead of temp file name
    video_name = Path(original_filename).stem
    
    # Construct ffmpeg command with additional options for large files
    output_pattern = os.path.join(temp_dir, f"frame_%06d.{output_format}")
    
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps=1/{interval_seconds}",  # Extract one frame every N seconds
        "-q:v", "2",  # High quality
        "-threads", "2",  # Limit threads to reduce memory usage
        "-y",  # Overwrite output files
        output_pattern
    ]
    
    try:
        # Run ffmpeg with timeout and better error handling
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=300  # 5 minute timeout
        )
        
        # Get list of extracted frames
        frame_files = sorted([f for f in os.listdir(temp_dir) if f.endswith(f'.{output_format}')])
        
        if not frame_files:
            st.error("No frames were extracted. The video might be corrupted or in an unsupported format.")
            return [], None
        
        # Rename files to include timestamps using original video name
        renamed_files = []
        for i, frame_file in enumerate(frame_files):
            timestamp = i * interval_seconds
            hours = timestamp // 3600
            minutes = (timestamp % 3600) // 60
            seconds = timestamp % 60
            
            timestamp_str = f"{hours:02d}h{minutes:02d}m{seconds:02d}s"
            new_name = f"{video_name}_{timestamp_str}.{output_format}"
            
            old_path = os.path.join(temp_dir, frame_file)
            new_path = os.path.join(temp_dir, new_name)
            
            try:
                os.rename(old_path, new_path)
                renamed_files.append(new_path)
            except OSError as e:
                st.warning(f"Could not rename {frame_file}: {e}")
                renamed_files.append(old_path)
        
        return renamed_files, temp_dir
        
    except subprocess.TimeoutExpired:
        st.error("Video processing timed out. Try using a larger interval or a smaller video file.")
        return [], None
    except subprocess.CalledProcessError as e:
        st.error(f"Error extracting frames: {e.stderr if e.stderr else 'Unknown ffmpeg error'}")
        return [], None
    except Exception as e:
        st.error(f"Unexpected error during frame extraction: {str(e)}")
        return [], None

def detect_and_blur_faces_and_plates(image_path, blur_faces=True, blur_plates=True):
    """
    Detect and blur faces and license plates using lightweight methods
    """
    image = cv2.imread(image_path)
    if image is None:
        return image_path
    
    blurred_image = image.copy()
    height, width = image.shape[:2]
    
    # Face detection using OpenCV's Haar Cascades
    if blur_faces:
        try:
            # Load face detection classifier
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Blur each detected face
            for (x, y, w, h) in faces:
                # Add some padding around the face
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(width, x + w + padding)
                y2 = min(height, y + h + padding)
                
                # Blur the face region
                roi = blurred_image[y1:y2, x1:x2]
                if roi.size > 0:  # Make sure ROI is not empty
                    roi = cv2.GaussianBlur(roi, (51, 51), 0)
                    blurred_image[y1:y2, x1:x2] = roi
                    
        except Exception as e:
            st.warning(f"Face detection failed: {e}")
    
    # License plate detection using EasyOCR
    if blur_plates:
        try:
            # Initialize EasyOCR reader
            reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have CUDA
            results = reader.readtext(image)
            
            for (bbox, text, confidence) in results:
                # Clean the detected text
                clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                
                # Heuristics to identify potential license plates
                is_potential_plate = (
                    len(clean_text) >= 4 and len(clean_text) <= 10 and  # Reasonable length
                    confidence > 0.4 and  # Reasonable confidence
                    bool(re.search(r'\d', clean_text)) and  # Contains numbers
                    bool(re.search(r'[A-Z]', clean_text)) and  # Contains letters
                    len(re.findall(r'\d', clean_text)) >= 1 and  # At least 1 digit
                    len(re.findall(r'[A-Z]', clean_text)) >= 1  # At least 1 letter
                )
                
                # Additional pattern matching for common formats
                plate_patterns = [
                    r'^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,3}$',  # General format
                    r'^[0-9]{1,4}[A-Z]{1,4}[0-9]{0,4}$',  # Mixed format
                    r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$',        # UK format AA00AAA
                    r'^[A-Z]{1}[0-9]{1,3}[A-Z]{3}$',      # UK format A000AAA
                ]
                
                pattern_match = any(re.match(pattern, clean_text) for pattern in plate_patterns)
                
                if is_potential_plate or pattern_match:
                    # Get bounding box coordinates
                    (top_left, top_right, bottom_right, bottom_left) = bbox
                    
                    # Calculate bounding rectangle
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    
                    x1 = int(min(x_coords)) - 15  # Add padding
                    y1 = int(min(y_coords)) - 15
                    x2 = int(max(x_coords)) + 15
                    y2 = int(max(y_coords)) + 15
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)
                    
                    # Blur the license plate area
                    if x2 > x1 and y2 > y1:  # Make sure we have a valid region
                        roi = blurred_image[y1:y2, x1:x2]
                        if roi.size > 0:
                            roi = cv2.GaussianBlur(roi, (41, 41), 0)
                            blurred_image[y1:y2, x1:x2] = roi
                    
        except Exception as e:
            st.warning(f"License plate detection failed: {e}")
    
    # Save blurred image
    blurred_path = image_path.replace('.', '_blurred.')
    cv2.imwrite(blurred_path, blurred_image)
    
    return blurred_path

def process_extracted_frames(frame_paths, blur_faces=True, blur_plates=True):
    """
    Process all extracted frames to blur faces and license plates
    """
    processed_paths = []
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, frame_path in enumerate(frame_paths):
        status_text.text(f"🔒 Processing frame {i+1}/{len(frame_paths)} for privacy...")
        
        try:
            blurred_path = detect_and_blur_faces_and_plates(
                frame_path, 
                blur_faces=blur_faces, 
                blur_plates=blur_plates
            )
            processed_paths.append(blurred_path)
        except Exception as e:
            st.warning(f"Failed to process {Path(frame_path).name}: {e}")
            processed_paths.append(frame_path)  # Keep original if processing fails
        
        # Update progress
        progress_bar.progress((i + 1) / len(frame_paths))
    
    status_text.text("✅ Privacy processing complete!")
    return processed_paths

def create_pdf(image_paths, video_name):
    """Create PDF with all images, one per page"""
    
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
    
    story = []
    styles = getSampleStyleSheet()
    
    for img_path in image_paths:
        # Get image name for title
        img_name = Path(img_path).name
        
        # Add image name as title
        title = Paragraph(img_name, styles['Title'])
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        # Add image
        img = Image.open(img_path)
        
        # Calculate size to fit on page while maintaining aspect ratio
        page_width, page_height = A4
        max_width = page_width - 2*inch
        max_height = page_height - 3*inch  # Leave space for title
        
        img_width, img_height = img.size
        aspect_ratio = img_width / img_height
        
        if img_width > max_width:
            img_width = max_width
            img_height = img_width / aspect_ratio
        
        if img_height > max_height:
            img_height = max_height
            img_width = img_height * aspect_ratio
        
        # Create ReportLab image
        rl_img = RLImage(img_path, width=img_width, height=img_height)
        story.append(rl_img)
        
        # Add page break except for last image
        if img_path != image_paths[-1]:
            story.append(Spacer(1, 0.5*inch))
    
    doc.build(story)
    pdf_buffer.seek(0)
    
    return pdf_buffer

def create_zip(file_paths):
    """Create ZIP file containing all images"""
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in file_paths:
            file_name = Path(file_path).name
            zip_file.write(file_path, file_name)
    
    zip_buffer.seek(0)
    return zip_buffer

def main():
    st.title("🎬 ROC Photo Extraction with Privacy Protection")
    st.markdown("Extract frames from video files at specified intervals with automatic face and license plate blurring")
    
    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=['mp4', 'mov', 'avi', 'mkv', 'wmv', 'flv', 'm4v'],
        help="Supported formats: MP4, MOV, AVI, MKV, WMV, FLV, M4V"
    )
    
    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.success(f"✅ Uploaded: {uploaded_file.name} ({file_size_mb:.2f} MB)")
        
        if file_size_mb > 100:
            st.warning("⚠️ Large file detected! Processing may take several minutes. Consider using a larger interval to reduce processing time.")
        
        MAX_FILE_SIZE_MB = 500
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"❌ File too large! Maximum supported size is {MAX_FILE_SIZE_MB}MB. Your file is {file_size_mb:.2f}MB.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            interval = st.select_slider(
                "Extract frame every:",
                options=[1, 2, 3, 5, 10, 15, 30, 60],
                value=5,
                format_func=lambda x: f"{x} second{'s' if x != 1 else ''}"
            )
        
        with col2:
            output_format = st.selectbox(
                "Image format:",
                options=['jpg', 'png'],
                index=0
            )
        
        # Privacy Options Section
        st.subheader("🔒 Privacy Protection")
        st.markdown("Automatically detect and blur sensitive content in extracted frames")
        
        col_privacy1, col_privacy2 = st.columns(2)
        
        with col_privacy1:
            blur_faces = st.checkbox(
                "👤 Blur faces",
                value=True,
                help="Automatically detect and blur human faces in extracted frames"
            )
        
        with col_privacy2:
            blur_plates = st.checkbox(
                "🚗 Blur license plates", 
                value=True,
                help="Automatically detect and blur license plates in extracted frames"
            )
        
        if blur_faces or blur_plates:
            st.info("💡 Privacy processing will add extra time to frame extraction but helps protect sensitive information")
        
        st.subheader("📥 Download Options")
        col3, col4 = st.columns(2)
        
        with col3:
            download_format = st.radio(
                "Download format:",
                options=['Individual images', 'ZIP file', 'PDF document'],
                index=0
            )
        
        with col4:
            if download_format == 'Individual images':
                st.info("💡 Individual download buttons will appear for each image")
            elif download_format == 'ZIP file':
                st.info("💡 All images will be packaged in a single ZIP file")
            elif download_format == 'PDF document':
                st.info("💡 All images will be combined into a PDF (one image per page)")
        
        if st.button("🎬 Extract Frames", type="primary"):
            with st.spinner("Extracting frames from video... This may take a few minutes for large files."):
                
                tmp_video_path = None
                temp_dir = None
                
                try:
                    # Save uploaded file to temp location without loading it all into memory
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_video:
                        tmp_video_path = tmp_video.name
                        # Use shutil.copyfileobj to write the file in chunks, avoiding high RAM usage
                        shutil.copyfileobj(uploaded_file, tmp_video)

                    st.info(f"📁 Temporary file created: {Path(tmp_video_path).name}")
                    
                    # Extract frames
                    extracted_files, temp_dir = extract_frames(tmp_video_path, interval, output_format, uploaded_file.name)
                    
                    if extracted_files:
                        st.success(f"✅ Extracted {len(extracted_files)} frames!")
                        
                        # Apply privacy filters if requested
                        if blur_faces or blur_plates:
                            with st.spinner("🔒 Applying privacy protection..."):
                                extracted_files = process_extracted_frames(
                                    extracted_files, 
                                    blur_faces=blur_faces, 
                                    blur_plates=blur_plates
                                )
                        
                        # --- Preview Section ---
                        st.subheader("👁️ Preview")
                        if blur_faces or blur_plates:
                            st.caption("Preview shows processed images with privacy protection applied")
                        
                        preview_cols = st.columns(min(3, len(extracted_files)))
                        for i, col in enumerate(preview_cols):
                            if i < len(extracted_files):
                                with col:
                                    try:
                                        img = Image.open(extracted_files[i])
                                        if img.size[0] > 800 or img.size[1] > 600:
                                            img.thumbnail((400, 300), Image.Resampling.LANCZOS)
                                        st.image(img, caption=Path(extracted_files[i]).name, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Could not display preview: {e}")
                        
                        if len(extracted_files) > 3:
                            st.info(f"... and {len(extracted_files) - 3} more images")
                        
                        # --- Download Section ---
                        st.subheader("📥 Download")
                        video_name = Path(uploaded_file.name).stem

                        if download_format == 'Individual images':
                            cols_per_row = 3
                            for i in range(0, len(extracted_files), cols_per_row):
                                cols = st.columns(cols_per_row)
                                for j, col in enumerate(cols):
                                    idx = i + j
                                    if idx < len(extracted_files):
                                        with col:
                                            with open(extracted_files[idx], 'rb') as f:
                                                file_name = Path(extracted_files[idx]).name
                                                st.download_button(
                                                    label=f"📥 {file_name}",
                                                    data=f.read(),
                                                    file_name=file_name,
                                                    mime=f"image/{output_format}",
                                                    key=f"download_{idx}"
                                                )
                        
                        elif download_format == 'ZIP file':
                            zip_buffer = create_zip(extracted_files)
                            suffix = "_privacy_protected" if (blur_faces or blur_plates) else ""
                            st.download_button(
                                label=f"📦 Download ZIP ({len(extracted_files)} images)",
                                data=zip_buffer,
                                file_name=f"{video_name}_frames{suffix}.zip",
                                mime="application/zip"
                            )
                        
                        elif download_format == 'PDF document':
                            with st.spinner("Creating PDF..."):
                                pdf_buffer = create_pdf(extracted_files, video_name)
                                suffix = "_privacy_protected" if (blur_faces or blur_plates) else ""
                                st.download_button(
                                    label=f"📄 Download PDF ({len(extracted_files)} images)",
                                    data=pdf_buffer,
                                    file_name=f"{video_name}_frames{suffix}.pdf",
                                    mime="application/pdf"
                                )
                    
                    else:
                        st.error("❌ Failed to extract frames. The video might be empty or processing failed.")

                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                
                finally:
                    # Cleanup code that runs regardless of success or failure
                    if tmp_video_path and os.path.exists(tmp_video_path):
                        os.unlink(tmp_video_path)
                        st.info("🗑️ Temporary video file cleaned up")
                    if temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        st.info("🗑️ Temporary image directory cleaned up")

# Password protection and main app
if __name__ == "__main__":
    # To run without password locally, comment out the 'if check_password():' and just call main()
    # To use password, ensure you have a secrets.toml file configured.
    # For example:
    # [passwords]
    # app_password = "your_secret_password_here"

    if check_password():
        main()
