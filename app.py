import streamlit as st
from google import genai
from google.genai import types
import yt_dlp
import tempfile
import os
import json
from typing import Dict, Optional

# Configure Gemini
def configure_gemini(api_key: str):
    """Configure Gemini API"""
    client = genai.Client(api_key=api_key)
    return client

def get_video_info_basic(url: str) -> Dict:
    """Get basic video info without downloading"""
    ydl_opts = {
        'skip_download': True,
        'quiet': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),
                'url': url,
                'platform': 'YouTube' if 'youtube.com' in url or 'youtu.be' in url else 'Other'
            }
    except Exception as e:
        st.error(f"Error extracting video info: {str(e)}")
        return {'title': 'Unknown', 'duration': 0, 'url': url, 'platform': 'Unknown'}

def download_video_for_gemini(url: str, max_size_mb: int = 20) -> Optional[str]:
    """Download video for Gemini analysis (under 20MB)"""
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    ydl_opts = {
        'outtmpl': f'{temp_dir}/%(title)s.%(ext)s',
        'format': f'worst[filesize<{max_size_mb}M]/worst',  # Get smallest file under 20MB
        'quiet': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
            
            # Check file size
            if os.path.exists(video_path):
                file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                if file_size_mb > max_size_mb:
                    st.warning(f"Video is {file_size_mb:.1f}MB (over {max_size_mb}MB limit). Using YouTube URL directly.")
                    os.remove(video_path)
                    return None
                return video_path
            else:
                st.error("Failed to download video file")
                return None
                
    except Exception as e:
        st.error(f"Error downloading video: {str(e)}")
        return None

def analyze_video_with_gemini(client, video_source, user_investments: Dict, is_youtube_url: bool = False) -> str:
    """Analyze video using Gemini API"""

    prompt = f"""
    Analyze this video for investment relevance based on the user's portfolio:

    USER'S INVESTMENTS:
    {json.dumps(user_investments, indent=2)}

    Please analyze both visual and audio content and provide:
    1. Relevance Score (0-100): How relevant is this content to their investments?
    2. Specific Matches: Which of their investments are mentioned or related?
    3. Key Insights: What specific information is relevant to their portfolio?
    4. Visual Elements: Any charts, graphs, or visual data shown?
    5. Action Items: Any suggested actions based on the content?
    6. Risk Level: Is this promoting risky strategies?
    7. Content Summary: Brief summary of what the video discusses

    Focus on:
    - Stock tickers, company names, sector mentions
    - Price movements, technical analysis
    - Market news and trends
    - Investment strategies and advice
    - Visual charts, graphs, or financial data

    Respond ONLY in valid JSON format:
    {{
        "relevance_score": 0-100,
        "specific_matches": ["list of matching investments"],
        "key_insights": ["list of relevant insights"],
        "visual_elements": ["charts/graphs seen"],
        "action_items": ["suggested actions"],
        "risk_assessment": "low/medium/high",
        "content_summary": "brief summary of video content",
        "summary": "brief summary of relevance to user's portfolio"
    }}
    """

    try:
        if is_youtube_url:
            # Use YouTube URL directly
            response = client.models.generate_content(
                model='models/gemini-2.5-flash',
                contents=types.Content(
                    parts=[
                        types.Part(
                            file_data=types.FileData(file_uri=video_source)
                        ),
                        types.Part(text=prompt)
                    ]
                )
            )
        else:
            # Use uploaded video file
            with open(video_source, 'rb') as video_file:
                video_bytes = video_file.read()

            response = client.models.generate_content(
                model='models/gemini-2.5-flash',
                contents=types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                data=video_bytes,
                                mime_type='video/mp4'
                            )
                        ),
                        types.Part(text=prompt)
                    ]
                )
            )

        return response.text

    except Exception as e:
        st.error(f"Error analyzing video with Gemini: {str(e)}")
        return None

def parse_gemini_response(response_text: str) -> Dict:
    """Parse Gemini JSON response"""
    try:
        # Clean response text to extract JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        # Remove any non-JSON text before and after
        response_text = response_text.strip()
        
        # Try to find JSON object
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_text = response_text[start_idx:end_idx]
            return json.loads(json_text)
        else:
            # Fallback: try parsing the whole response
            return json.loads(response_text)
            
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON response: {str(e)}")
        st.error(f"Raw response: {response_text[:500]}...")
        return None

def display_analysis_results(analysis_result: Dict):
    """Display the analysis results in a nice format"""
    if not analysis_result:
        st.error("No analysis results to display")
        return
    
    # Relevance score with color coding
    score = analysis_result.get('relevance_score', 0)
    if score >= 80:
        st.success(f" Highly Relevant - Score: {score}/100")
    elif score >= 50:
        st.warning(f" Moderately Relevant - Score: {score}/100")
    else:
        st.info(f" Low Relevance - Score: {score}/100")
    
    # Content Summary
    if analysis_result.get('content_summary'):
        st.write(f"Content: {analysis_result['content_summary']}")
    
    # Summary
    st.write(f"Relevance: {analysis_result.get('summary', 'No summary available')}")
    
    # Detailed breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        if analysis_result.get('specific_matches'):
            st.subheader(" Matching Investments")
            for match in analysis_result['specific_matches']:
                st.write(f"- {match}")
        
        if analysis_result.get('key_insights'):
            st.subheader(" Key Insights")
            for insight in analysis_result['key_insights']:
                st.write(f"- {insight}")
    
    with col2:
        if analysis_result.get('action_items'):
            st.subheader(" Suggested Actions")
            for action in analysis_result['action_items']:
                st.write(f"- {action}")
        
        # Risk assessment
        risk = analysis_result.get('risk_assessment', 'unknown').lower()
        if risk == 'high':
            st.error(f" Risk Level: {risk.upper()}")
        elif risk == 'medium':
            st.warning(f" Risk Level: {risk.upper()}")
        else:
            st.success(f" Risk Level: {risk.upper()}")
        
        # Visual elements if any
        if analysis_result.get('visual_elements'):
            st.subheader(" Visual Elements")
            for element in analysis_result['visual_elements']:
                st.write(f"- {element}")

def main():
    st.set_page_config(
        page_title="Investment Video Analyzer",
        layout="wide"
    )
    
    st.title(" Investment Video Analyzer")
    st.markdown("Analyze videos for investment relevance using Google's Gemini AI")
    
    # Sidebar for API key and investment details
    with st.sidebar:
        st.header(" Configuration")
        
        # Gemini API Key
        api_key = st.text_input("Gemini API Key", type="password")
        
        if not api_key:
            st.warning("Please enter your Gemini API key to continue")
            st.info("Get your API key from: [Google AI Studio](https://aistudio.google.com/app/apikey)")
            st.stop()
        
        # Configure Gemini
        try:
            client = configure_gemini(api_key)
            st.success(" Gemini API connected")
        except Exception as e:
            st.error(f" Failed to connect to Gemini: {str(e)}")
            st.stop()
        
        st.header(" Your Investment Portfolio")
        
        # Investment details form
        with st.form("investment_form"):
            st.subheader("Add Investment Details")
            
            investment_type = st.selectbox(
                "Investment Type",
                ["Stock", "Cryptocurrency", "ETF", "Mutual Fund", "Bond", "Commodity"]
            )
            
            symbol = st.text_input("Symbol/Ticker (e.g., AAPL, BTC)")
            amount = st.number_input("Investment Amount ($)", min_value=0.0)
            notes = st.text_area("Additional Notes")
            
            submitted = st.form_submit_button("Add Investment")
            
            if submitted and symbol:
                if 'investments' not in st.session_state:
                    st.session_state.investments = {}
                
                st.session_state.investments[symbol] = {
                    'type': investment_type,
                    'amount': amount,
                    'notes': notes
                }
                st.success(f"Added {symbol} to portfolio")
        
        # Display current investments
        if 'investments' in st.session_state and st.session_state.investments:
            st.subheader(" Current Portfolio")
            for symbol, details in st.session_state.investments.items():
                with st.expander(f"{symbol} - {details['type']}"):
                    st.write(f"Amount: ${details['amount']:,.2f}")
                    if details['notes']:
                        st.write(f"Notes: {details['notes']}")
                    if st.button(f"Remove {symbol}", key=f"remove_{symbol}"):
                        del st.session_state.investments[symbol]
                        st.rerun()
    
    # Main content area
    if 'investments' not in st.session_state or not st.session_state.investments:
        st.warning(" Please add your investment details in the sidebar to get started")
        st.stop()
    
    # Video input section
    st.header(" Video Analysis")
    
    input_type = st.radio(
        "Select input type:",
        ["YouTube URL", "Instagram/TikTok/Other URL", "Upload Video File"]
    )
    
    if input_type == "YouTube URL":
        video_url = st.text_input("Enter YouTube URL:")
        
        if video_url and st.button("Analyze YouTube Video"):
            with st.spinner("Analyzing video with Gemini..."):
                
                # Get basic video info
                video_info = get_video_info_basic(video_url)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(" Video Information")
                    st.write(f"Title: {video_info['title']}")
                    st.write(f"Duration: {video_info['duration']} seconds")
                    st.write(f"Platform: {video_info['platform']}")
                
                with col2:
                    st.subheader(" Analysis Method")
                    st.write(" Using YouTube URL directly with Gemini")
                    st.write("- Full video and audio analysis")
                    st.write("- Visual element detection")
                    st.write("- Investment relevance scoring")
                
                # Analyze with Gemini
                st.info(" Analyzing video content...")
                response_text = analyze_video_with_gemini(
                    client, video_url, st.session_state.investments, is_youtube_url=True
                )
                
                if response_text:
                    analysis_result = parse_gemini_response(response_text)
                    
                    if analysis_result:
                        st.header(" Analysis Results")
                        display_analysis_results(analysis_result)
                    else:
                        st.error("Failed to parse analysis results")
                        with st.expander("View Raw Response"):
                            st.text(response_text)
    
    elif input_type == "Instagram/TikTok/Other URL":
        video_url = st.text_input("Enter video URL:")
        
        if video_url and st.button("Analyze Video"):
            with st.spinner("Downloading and analyzing video..."):
                
                # Get basic video info
                video_info = get_video_info_basic(video_url)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(" Video Information")
                    st.write(f"Title: {video_info['title']}")
                    st.write(f"Duration: {video_info['duration']} seconds")
                    st.write(f"Platform: {video_info['platform']}")
                
                with col2:
                    st.subheader(" Analysis Method")
                    st.write(" Download + Gemini Analysis")
                    st.write("- Download video file")
                    st.write("- Upload to Gemini for analysis")
                    st.write("- Full content processing")
                
                # Download video
                st.info(" Downloading video...")
                video_path = download_video_for_gemini(video_url)
                
                if video_path:
                    try:
                        # Show file info
                        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                        st.success(f" Downloaded video ({file_size_mb:.1f} MB)")
                        
                        # Analyze with Gemini
                        st.info(" Analyzing video content...")
                        response_text = analyze_video_with_gemini(
                            client, video_path, st.session_state.investments, is_youtube_url=False
                        )
                        
                        if response_text:
                            analysis_result = parse_gemini_response(response_text)
                            
                            if analysis_result:
                                st.header(" Analysis Results")
                                display_analysis_results(analysis_result)
                            else:
                                st.error("Failed to parse analysis results")
                                with st.expander("View Raw Response"):
                                    st.text(response_text)
                        
                        # Cleanup
                        os.remove(video_path)
                        
                    except Exception as e:
                        st.error(f"Error processing video: {str(e)}")
                        if video_path and os.path.exists(video_path):
                            os.remove(video_path)
    
    elif input_type == "Upload Video File":
        uploaded_file = st.file_uploader(
            "Choose a video file", 
            type=['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'],
            help="Maximum file size: 20MB"
        )
        
        if uploaded_file is not None:
            # Check file size
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(" File Information")
                st.write(f"Filename: {uploaded_file.name}")
                st.write(f"Size: {file_size_mb:.1f} MB")
                st.write(f"Type: {uploaded_file.type}")
            
            with col2:
                st.subheader(" Analysis Method")
                st.write(" Direct Upload to Gemini")
                st.write("- Process uploaded file")
                st.write("- Full video analysis")
                st.write("- Investment relevance scoring")
            
            if file_size_mb > 20:
                st.error(" File size exceeds 20MB limit. Please upload a smaller file.")
            elif st.button("Analyze Uploaded Video"):
                with st.spinner("Analyzing uploaded video..."):
                    
                    # Save uploaded file temporarily
                    temp_dir = tempfile.mkdtemp()
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())
                    
                    try:
                        # Analyze with Gemini
                        st.info(" Analyzing video content...")
                        response_text = analyze_video_with_gemini(
                            client, temp_path, st.session_state.investments, is_youtube_url=False
                        )
                        
                        if response_text:
                            analysis_result = parse_gemini_response(response_text)
                            
                            if analysis_result:
                                st.header(" Analysis Results")
                                display_analysis_results(analysis_result)
                            else:
                                st.error("Failed to parse analysis results")
                                with st.expander("View Raw Response"):
                                    st.text(response_text)
                        
                        # Cleanup
                        os.remove(temp_path)
                        os.rmdir(temp_dir)
                        
                    except Exception as e:
                        st.error(f"Error analyzing video: {str(e)}")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        if os.path.exists(temp_dir):
                            os.rmdir(temp_dir)
    


if __name__ == "__main__":
    main()
