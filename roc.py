import streamlit as st
import tempfile
import os
import subprocess
import zipfile
from pathlib import Path
from PIL import Image
from reportlab.lib.pagesizes import letter, A4, A3
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

def reset_session_state():
    """Reset all processing-related session state variables"""
    keys_to_reset = [
        'processed_data', 
        'silent_video_data', 
        'processing_complete', 
        'last_uploaded_file'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

def remove_audio_from_video(input_path, original_filename):
    """Remove audio from video using ffmpeg"""
    
    # Create output filename with "_processed" suffix
    video_stem = Path(original_filename).stem
    video_ext = Path(original_filename).suffix
    output_filename = f"{video_stem}_processed{video_ext}"
    
    # Create temporary file for output
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=video_ext)
    temp_output.close()
    
    # Construct ffmpeg command to remove audio
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "copy",  # Copy video stream without re-encoding
        "-an",  # Remove audio stream
        "-y",  # Overwrite output files
        temp_output.name
    ]
    
    try:
        # Run ffmpeg with timeout
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=300  # 5 minute timeout
        )
        
        return temp_output.name, output_filename
        
    except subprocess.TimeoutExpired:
        st.error("Audio removal timed out. The video file might be too large.")
        if os.path.exists(temp_output.name):
            os.unlink(temp_output.name)
        return None, None
    except subprocess.CalledProcessError as e:
        st.error(f"Error removing audio: {e.stderr if e.stderr else 'Unknown ffmpeg error'}")
        if os.path.exists(temp_output.name):
            os.unlink(temp_output.name)
        return None, None
    except Exception as e:
        st.error(f"Unexpected error during audio removal: {str(e)}")
        if os.path.exists(temp_output.name):
            os.unlink(temp_output.name)
        return None, None

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

def get_page_size(paper_size, orientation):
    """Get page dimensions based on paper size and orientation"""
    if paper_size == "A4":
        base_size = A4
    else:  # A3
        base_size = A3
    
    if orientation == "Portrait":
        return base_size  # (width, height)
    else:  # Landscape
        return (base_size[1], base_size[0])  # Swap width and height

def create_pdf(image_paths, tower_number, video_filename, paper_size="A4", orientation="Portrait"):
    """Create PDF with all images, one per page, centered and filling 80% of page"""
    
    pdf_buffer = io.BytesIO()
    page_size = get_page_size(paper_size, orientation)
    doc = SimpleDocTemplate(pdf_buffer, pagesize=page_size)
    
    story = []
    styles = getSampleStyleSheet()
    
    # Calculate available space (80% of page)
    page_width, page_height = page_size
    margin_width = page_width * 0.1  # 10% margin on each side
    margin_height = page_height * 0.1  # 10% margin on top and bottom
    
    max_width = page_width * 0.8  # 80% of page width
    max_height = page_height * 0.8  # 80% of page height
    
    for i, img_path in enumerate(image_paths):
        # Extract timestamp from filename
        img_name = Path(img_path).name
        # Extract timestamp (assumes format: videoname_XXhYYmZZs.ext)
        timestamp = "Unknown"
        if '_' in img_name:
            timestamp_part = img_name.split('_')[-1].split('.')[0]  # Get part before extension
            if 'h' in timestamp_part and 'm' in timestamp_part and 's' in timestamp_part:
                timestamp = timestamp_part
        
        # Add some top spacing to center content vertically
        story.append(Spacer(1, margin_height * 0.2))
        
        # Create three-line title
        title_text = f"""ROC Photos For {tower_number}<br/>
Video File: {video_filename}<br/>
Timestamp: {timestamp}"""
        
        title = Paragraph(title_text, styles['Normal'])
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        # Load image and calculate optimal size
        img = Image.open(img_path)
        img_width, img_height = img.size
        aspect_ratio = img_width / img_height
        
        # Calculate the size to fill 80% of page while maintaining aspect ratio
        # Account for title space (approximately 1 inch)
        available_height = max_height - 1*inch
        
        if aspect_ratio > (max_width / available_height):
            # Image is wider relative to available space - fit to width
            final_width = max_width
            final_height = final_width / aspect_ratio
        else:
            # Image is taller relative to available space - fit to height
            final_height = available_height
            final_width = final_height * aspect_ratio
        
        # Create ReportLab image - it will be centered automatically by SimpleDocTemplate
        rl_img = RLImage(img_path, width=final_width, height=final_height)
        story.append(rl_img)
        
        # Add remaining space to center content vertically
        if i < len(image_paths) - 1:  # Not the last image
            remaining_space = available_height - final_height
            if remaining_space > 0:
                story.append(Spacer(1, remaining_space * 0.4))
            
            # Force page break
            from reportlab.platypus import PageBreak
            story.append(PageBreak())
    
    # Build PDF with custom margins to center content
    doc.leftMargin = margin_width
    doc.rightMargin = margin_width
    doc.topMargin = margin_height * 0.6
    doc.bottomMargin = margin_height * 0.6
    
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
    
    # Initialize session state for persistent data
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'silent_video_data' not in st.session_state:
        st.session_state.silent_video_data = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    
    # Add reset button in the sidebar or at the top
    if st.session_state.processing_complete:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔄 Reset")
        if st.sidebar.button("🔄 Reset Application", type="secondary", help="Clear all data and start over"):
            reset_session_state()
            st.rerun()
    
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
        
        # Audio removal option
        st.subheader("🔇 Audio Options")
        remove_audio_option = st.checkbox(
            "Remove audio from video",
            help="Create a processed version of the video without audio alongside frame extraction"
        )
        
        if remove_audio_option:
            st.info("📹 A processed video file without audio will be available for download after processing")
        
        st.subheader("🖼️ Frame Extraction Settings")
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
        
        st.subheader("📦 Download Options")
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
        
        # Tower number input (only show when PDF is selected)
        if download_format == 'PDF document':
            tower_number = st.text_input(
                "Tower Number(s):",
                placeholder="e.g., CB4-8B",
                help="This will be used in the PDF filename and image titles"
            )
        
        # PDF-specific options
        if download_format == 'PDF document':
            st.subheader("📄 PDF Options")
            pdf_col1, pdf_col2 = st.columns(2)
            
            with pdf_col1:
                paper_size = st.selectbox(
                    "Paper size:",
                    options=['A4', 'A3'],
                    index=0
                )
            
            with pdf_col2:
                orientation = st.selectbox(
                    "Orientation:",
                    options=['Portrait', 'Landscape'],
                    index=0
                )
            
            st.info(f"📄 PDF will be created in {paper_size} {orientation.lower()} format with images centered and filling 80% of the page")
        
        # Reset processing when a new file is uploaded
        if uploaded_file and st.session_state.get('last_uploaded_file') != uploaded_file.name:
            st.session_state.processed_data = None
            st.session_state.silent_video_data = None
            st.session_state.processing_complete = False
            st.session_state.last_uploaded_file = uploaded_file.name

        if st.button("🎬 Process Video", type="primary") or st.session_state.processing_complete:
            if not st.session_state.processing_complete:
                # Only process if we haven't already processed
                with st.spinner("Processing video... This may take a few minutes for large files."):
                    
                    tmp_video_path = None
                    temp_dir = None
                    silent_video_path = None
                    
                    try:
                        # Save uploaded file to temp location without loading it all into memory
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_video:
                            tmp_video_path = tmp_video.name
                            # Use shutil.copyfileobj to write the file in chunks, avoiding high RAM usage
                            shutil.copyfileobj(uploaded_file, tmp_video)

                        st.info(f"📁 Temporary file created: {Path(tmp_video_path).name}")
                        
                        # Remove audio if requested
                        silent_video_data = None
                        silent_video_filename = None
                        if remove_audio_option:
                            with st.spinner("Removing audio from video..."):
                                silent_video_path, silent_video_filename = remove_audio_from_video(tmp_video_path, uploaded_file.name)
                                if silent_video_path:
                                    st.success("✅ Audio removed successfully!")
                                    # Store video data in session state
                                    with open(silent_video_path, 'rb') as f:
                                        silent_video_data = f.read()
                                else:
                                    st.error("❌ Failed to remove audio from video")
                        
                        # Extract frames
                        with st.spinner("Extracting frames..."):
                            extracted_files, temp_dir = extract_frames(tmp_video_path, interval, output_format, uploaded_file.name)
                        
                        if extracted_files:
                            st.success(f"✅ Extracted {len(extracted_files)} frames!")
                            
                            # Store all processed data in session state
                            processed_data = {
                                'extracted_files': [],
                                'video_name': Path(uploaded_file.name).stem,
                                'original_filename': uploaded_file.name,
                                'output_format': output_format,
                                'download_format': download_format,
                                'remove_audio_option': remove_audio_option
                            }
                            
                            # Copy extracted files data to session state
                            for file_path in extracted_files:
                                with open(file_path, 'rb') as f:
                                    file_data = f.read()
                                    processed_data['extracted_files'].append({
                                        'name': Path(file_path).name,
                                        'data': file_data
                                    })
                            
                            # Store in session state
                            st.session_state.processed_data = processed_data
                            if silent_video_data:
                                st.session_state.silent_video_data = {
                                    'data': silent_video_data,
                                    'filename': silent_video_filename
                                }
                            st.session_state.processing_complete = True
                            
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
                        if silent_video_path and os.path.exists(silent_video_path):
                            os.unlink(silent_video_path)
                            st.info("🗑️ Processed video file cleaned up")

            # Display results from session state
            if st.session_state.processed_data:
                processed_data = st.session_state.processed_data
                
                # --- Preview Section ---
                st.subheader("🖼️ Preview")
                preview_cols = st.columns(min(3, len(processed_data['extracted_files'])))
                for i, col in enumerate(preview_cols):
                    if i < len(processed_data['extracted_files']):
                        with col:
                            try:
                                img_data = processed_data['extracted_files'][i]['data']
                                img = Image.open(io.BytesIO(img_data))
                                if img.size[0] > 800 or img.size[1] > 600:
                                    img.thumbnail((400, 300), Image.Resampling.LANCZOS)
                                st.image(img, caption=processed_data['extracted_files'][i]['name'], use_container_width=True)
                            except Exception as e:
                                st.error(f"Could not display preview: {e}")
                
                if len(processed_data['extracted_files']) > 3:
                    st.info(f"... and {len(processed_data['extracted_files']) - 3} more images")
                
                # --- Download Section ---
                st.subheader("📥 Downloads")
                
                # Video download section (if audio removal was requested)
                if st.session_state.silent_video_data:
                    st.markdown("#### 📹 Video (Processed)")
                    silent_data = st.session_state.silent_video_data
                    st.download_button(
                        label=f"📹 Download {silent_data['filename']}",
                        data=silent_data['data'],
                        file_name=silent_data['filename'],
                        mime="video/mp4",
                        help="Processed video with audio removed",
                        key="download_processed_video"
                    )
                
                # Images download section
                st.markdown("#### 🖼️ Extracted Images")

                if download_format == 'Individual images':
                    cols_per_row = 3
                    for i in range(0, len(processed_data['extracted_files']), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            idx = i + j
                            if idx < len(processed_data['extracted_files']):
                                with col:
                                    file_info = processed_data['extracted_files'][idx]
                                    st.download_button(
                                        label=f"📥 {file_info['name']}",
                                        data=file_info['data'],
                                        file_name=file_info['name'],
                                        mime=f"image/{processed_data['output_format']}",
                                        key=f"download_img_{idx}"
                                    )
                
                elif download_format == 'ZIP file':
                    # Create ZIP from session state data
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for file_info in processed_data['extracted_files']:
                            zip_file.writestr(file_info['name'], file_info['data'])
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label=f"📦 Download ZIP ({len(processed_data['extracted_files'])} images)",
                        data=zip_buffer,
                        file_name=f"{processed_data['video_name']}_frames.zip",
                        mime="application/zip",
                        key="download_zip"
                    )
                
                elif download_format == 'PDF document':
                    if not tower_number or tower_number.strip() == "":
                        st.error("❌ Please enter a Tower Number for PDF generation.")
                    else:
                        # Create temporary files for PDF generation
                        with tempfile.TemporaryDirectory() as temp_pdf_dir:
                            temp_image_paths = []
                            for i, file_info in enumerate(processed_data['extracted_files']):
                                temp_path = os.path.join(temp_pdf_dir, file_info['name'])
                                with open(temp_path, 'wb') as f:
                                    f.write(file_info['data'])
                                temp_image_paths.append(temp_path)
                            
                            pdf_buffer = create_pdf(temp_image_paths, tower_number.strip(), processed_data['original_filename'], paper_size, orientation)
                            pdf_filename = f"ROC photos for {tower_number.strip()}.pdf"
                            st.download_button(
                                label=f"📄 Download PDF ({len(processed_data['extracted_files'])} images) - {paper_size} {orientation}",
                                data=pdf_buffer,
                                file_name=pdf_filename,
                                mime="application/pdf",
                                key="download_pdf"
                            )
                
                # Reset button in main area after downloads
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🔄 Reset and Process New Video", type="secondary", help="Clear all data and start over with a new video"):
                        reset_session_state()
                        st.rerun()


# Password protection and main app
if __name__ == "__main__":
    # To run without password locally, comment out the 'if check_password():' and just call main()
    # To use password, ensure you have a secrets.toml file configured.
    # For example:
    # [passwords]
    # app_password = "your_secret_password_here"

    if check_password():
        main()
