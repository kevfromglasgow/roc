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
    
    # Construct ffmpeg command
    output_pattern = os.path.join(temp_dir, f"frame_%06d.{output_format}")
    
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps=1/{interval_seconds}",  # Extract one frame every N seconds
        "-y",  # Overwrite output files
        output_pattern
    ]
    
    try:
        # Run ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Get list of extracted frames
        frame_files = sorted([f for f in os.listdir(temp_dir) if f.endswith(f'.{output_format}')])
        
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
            
            os.rename(old_path, new_path)
            renamed_files.append(new_path)
        
        return renamed_files, temp_dir
        
    except subprocess.CalledProcessError as e:
        st.error(f"Error extracting frames: {e.stderr}")
        return [], None

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
    st.title("🎬 ROC Photo Extraction")
    st.markdown("Extract frames from video files at specified intervals")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=['mp4', 'mov', 'avi', 'mkv', 'wmv', 'flv', 'm4v'],
        help="Supported formats: MP4, MOV, AVI, MKV, WMV, FLV, M4V"
    )
    
    if uploaded_file is not None:
        # Display video info
        st.success(f"✅ Uploaded: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.2f} MB)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Interval selection
            interval = st.select_slider(
                "Extract frame every:",
                options=[1, 2, 3, 5, 10, 15, 30, 60],
                value=5,
                format_func=lambda x: f"{x} second{'s' if x != 1 else ''}"
            )
        
        with col2:
            # Output format selection
            output_format = st.selectbox(
                "Image format:",
                options=['jpg', 'png'],
                index=0
            )
        
        # Download options
        st.subheader("Download Options")
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
        
        # Extract frames button
        if st.button("🎬 Extract Frames", type="primary"):
            with st.spinner("Extracting frames from video..."):
                
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_video:
                    tmp_video.write(uploaded_file.getvalue())
                    tmp_video_path = tmp_video.name
                
                # Extract frames - pass original filename
                extracted_files, temp_dir = extract_frames(tmp_video_path, interval, output_format, uploaded_file.name)
                
                # Clean up temporary video file
                os.unlink(tmp_video_path)
                
                if extracted_files:
                    st.success(f"✅ Extracted {len(extracted_files)} frames!")
                    
                    # Show preview of first few images
                    st.subheader("Preview")
                    
                    # Display first 3 images as preview - FIXED: use_container_width instead of use_column_width
                    preview_cols = st.columns(min(3, len(extracted_files)))
                    for i, col in enumerate(preview_cols):
                        if i < len(extracted_files):
                            with col:
                                img = Image.open(extracted_files[i])
                                st.image(img, caption=Path(extracted_files[i]).name, use_container_width=True)
                    
                    if len(extracted_files) > 3:
                        st.info(f"... and {len(extracted_files) - 3} more images")
                    
                    # Download section
                    st.subheader("Download")
                    
                    video_name = Path(uploaded_file.name).stem
                    
                    if download_format == 'Individual images':
                        st.write("Download individual images:")
                        
                        # Create columns for download buttons
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
                        st.download_button(
                            label=f"📦 Download ZIP ({len(extracted_files)} images)",
                            data=zip_buffer,
                            file_name=f"{video_name}_frames.zip",
                            mime="application/zip"
                        )
                    
                    elif download_format == 'PDF document':
                        with st.spinner("Creating PDF..."):
                            pdf_buffer = create_pdf(extracted_files, video_name)
                            st.download_button(
                                label=f"📄 Download PDF ({len(extracted_files)} images)",
                                data=pdf_buffer,
                                file_name=f"{video_name}_frames.pdf",
                                mime="application/pdf"
                            )
                    
                    # Clean up temporary directory
                    shutil.rmtree(temp_dir, ignore_errors=True)
                
                else:
                    st.error("❌ Failed to extract frames from video")

# Password protection and main app
if check_password():
    main()
