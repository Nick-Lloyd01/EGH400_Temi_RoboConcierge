# Video Processing Guide

Complete guide for processing pre-recorded videos with the Temi Face Detection system.

---

## 🎥 Quick Start

### 1. Enable Video Mode
Edit `config.py`:
```python
VIDEO_MODE = True
INPUT_VIDEO_PATH = 'InputVideos/my_video.MOV'
OUTPUT_VIDEO_PATH = 'ResultsVideos/output.mp4'
```

### 2. Run Processing
```bash
python main.py
```

### 3. Get Your Output
Processed video saved to: `ResultsVideos/output.mp4`

---

## ⚙️ Configuration Options

### Basic Setup
```python
# Enable video processing mode
VIDEO_MODE = True

# Input video path (relative or absolute)
INPUT_VIDEO_PATH = 'InputVideos/my_video.MOV'

# Output video path
OUTPUT_VIDEO_PATH = 'ResultsVideos/analyzed_output.mp4'
```

### Display Options
```python
# Show preview window while processing (slower)
SHOW_DISPLAY_WINDOW = True

# No preview window (faster processing - 10-30% speed boost)
SHOW_DISPLAY_WINDOW = False
```

### Advanced Settings
```python
# Video codec ('mp4v', 'XVID', 'H264', 'avc1')
OUTPUT_VIDEO_CODEC = 'mp4v'

# Output FPS (None = match input video)
OUTPUT_VIDEO_FPS = None

# Or specify custom FPS
OUTPUT_VIDEO_FPS = 30
```

---

## 📁 File Paths

### Relative Paths (Recommended)
```python
INPUT_VIDEO_PATH = 'InputVideos/video.MOV'
OUTPUT_VIDEO_PATH = 'ResultsVideos/output.mp4'
```

### Absolute Paths
```python
INPUT_VIDEO_PATH = '/Users/username/Desktop/videos/input.mp4'
OUTPUT_VIDEO_PATH = '/Users/username/Desktop/results/output.mp4'
```

### Supported Formats
- **Input**: `.mp4`, `.mov`, `.avi`, `.mkv`, `.MOV`
- **Output**: `.mp4` (recommended), `.avi`

---

## 📊 What Gets Saved

The output video includes all visual overlays:

### Detection Overlays
- ✅ Face bounding boxes (cyan)
- ✅ 468 facial landmarks (green dots)
- ✅ 3D head pose axes (RGB arrows)
- ✅ Person boxes when detected (yellow)

### Information Display

**Top Right Corner:**
- Face confidence score
- Yaw angle (degrees)
- Pitch angle (degrees)
- Roll angle (degrees)
- Distance estimate (meters)

**Top Left Corner:**
- Gaze status (LOOKING / NOT LOOKING)
- Sustained gaze timer
- Approach behavior triggers

**Bottom Left Corner:**
- Active subsumption layer
- All layer statuses
- Suppressed layer indicators

---

## 🚀 Processing Modes

### With Display Preview (Default)
```python
VIDEO_MODE = True
SHOW_DISPLAY_WINDOW = True
```
- Shows live preview window
- Watch processing in real-time
- Slightly slower due to rendering
- Can monitor progress visually

### Fast Processing (No Display)
```python
VIDEO_MODE = True
SHOW_DISPLAY_WINDOW = False
```
- No preview window
- 10-30% faster processing
- Only console progress updates
- Ideal for batch processing

---

## 📝 Example Configurations

### Standard Processing
```python
VIDEO_MODE = True
SHOW_DISPLAY_WINDOW = True
INPUT_VIDEO_PATH = 'InputVideos/IMG_1322.MOV'
OUTPUT_VIDEO_PATH = 'ResultsVideos/IMG_1322_analyzed.mp4'
OUTPUT_VIDEO_CODEC = 'mp4v'
OUTPUT_VIDEO_FPS = None  # Match input FPS
```

### Fast Batch Processing
```python
VIDEO_MODE = True
SHOW_DISPLAY_WINDOW = False  # No display for speed
INPUT_VIDEO_PATH = 'InputVideos/batch_video.mp4'
OUTPUT_VIDEO_PATH = 'ResultsVideos/batch_output.mp4'
OUTPUT_VIDEO_CODEC = 'XVID'  # Fast encoding
```

### High Quality Output
```python
VIDEO_MODE = True
SHOW_DISPLAY_WINDOW = True
INPUT_VIDEO_PATH = 'InputVideos/high_res.MOV'
OUTPUT_VIDEO_PATH = 'ResultsVideos/high_quality.mp4'
OUTPUT_VIDEO_CODEC = 'H264'  # Better quality
OUTPUT_VIDEO_FPS = 30  # Standard FPS
```

---

## 🎬 Processing Workflow

### What Happens During Processing

1. **Initialization**
   ```
   📹 Opening video file: InputVideos/video.MOV
   ✓ Video opened successfully
     Resolution: 1920x1080
     FPS: 30.0
     Total frames: 1500
   ✓ Video writer initialized
   ```

2. **Processing**
   - Reads frame from input video
   - Detects face with YOLO
   - Extracts facial landmarks (MediaPipe)
   - Estimates 3D head pose
   - Analyzes gaze direction
   - Runs behavior management
   - Renders all overlays
   - Writes frame to output video

3. **Completion**
   ```
   ✅ Video processing complete - End of file reached
   💾 Output video saved: ResultsVideos/output.mp4
   📊 Average FPS: 28.5
   📊 Total frames processed: 1500
   ```

---

## ⌨️ Controls During Processing

- **'q'** - Stop processing and save current progress
- **'r'** - Reset behavior manager state
- **'s'** - Print detailed status to console
- **'p'** - Toggle performance logging

Press **'q'** to stop early - output video will be saved with frames processed so far.

---

## 🔄 Batch Processing Multiple Videos

Create a processing script:

```python
# process_batch.py
import config
import main

videos = [
    ('InputVideos/video1.MOV', 'ResultsVideos/output1.mp4'),
    ('InputVideos/video2.MOV', 'ResultsVideos/output2.mp4'),
    ('InputVideos/video3.MOV', 'ResultsVideos/output3.mp4'),
]

# Fast processing mode
config.VIDEO_MODE = True
config.SHOW_DISPLAY_WINDOW = False

for input_path, output_path in videos:
    config.INPUT_VIDEO_PATH = input_path
    config.OUTPUT_VIDEO_PATH = output_path
    
    print(f"\n{'='*60}")
    print(f"Processing: {input_path}")
    print(f"{'='*60}")
    
    main.main()

print("\n✅ All videos processed!")
```

Run:
```bash
python process_batch.py
```

---

## 🐛 Troubleshooting

### Video File Not Found
```
❌ ERROR: Could not open video file: InputVideos/video.mov
```
**Solutions:**
- Check file path is correct
- Verify file exists in InputVideos folder
- Try absolute path
- Check file extension matches actual file

### Video Writer Fails
```
⚠ Warning: Could not initialize video writer
```
**Solutions:**
- Try different codec: `OUTPUT_VIDEO_CODEC = 'XVID'`
- Ensure output directory exists (auto-created normally)
- Check disk space
- Verify write permissions
- Use `.mp4` or `.avi` extension

### Slow Processing
**Normal Speed:** 10-25 FPS (slower than real-time)

**To Speed Up:**
1. Set `SHOW_DISPLAY_WINDOW = False`
2. Use codec `XVID` instead of `mp4v`
3. Close other applications
4. Reduce input video resolution before processing

### Output Video Issues
**Problem:** Output larger than input  
**Reason:** Additional rendering (overlays, text, graphics)  
**Solution:** Normal behavior, use H264 codec for better compression

**Problem:** Output quality poor  
**Solution:** Try different codec or ensure input quality is good

---

## 📈 Performance Expectations

### Processing Speed
- **Live Camera:** 20-30 FPS (real-time)
- **Video with Display:** 15-25 FPS
- **Video without Display:** 20-30 FPS

### File Sizes
Output videos typically **larger** than input due to:
- Rendered overlays (boxes, landmarks, text)
- Additional graphics (pose axes, gaze arrows)
- Codec compression settings

### Quality
- Same resolution as input video
- Same FPS as input (or custom if specified)
- Compression quality depends on codec used

---

## 💡 Tips & Best Practices

### For Best Results
1. **Good Lighting:** Ensure faces are well-lit in videos
2. **Face Visibility:** Keep faces visible and unobstructed
3. **Distance:** 1-3 meters optimal for detection
4. **Resolution:** Higher resolution = better detection

### File Management
1. **Organize Inputs:** Keep videos in `InputVideos/` folder
2. **Name Outputs:** Use descriptive output names
3. **Backup Originals:** Keep original videos safe
4. **Clean Up:** Remove old processed videos periodically

### Processing Strategy
1. **Test First:** Process short clip to verify settings
2. **Batch Similar:** Process similar videos together
3. **No Display:** Use `SHOW_DISPLAY_WINDOW = False` for batches
4. **Monitor:** Check console output for errors

---

## 🔙 Switching Back to Live Camera

```python
# In config.py
VIDEO_MODE = False  # That's it!
```

```bash
python main.py
```

All video settings are ignored in live camera mode.

---

## ❓ FAQ

**Q: Can I process videos while using live camera?**  
A: No, choose one mode at a time via `VIDEO_MODE` setting.

**Q: Does the output include audio?**  
A: No, only video with visual overlays. Audio is not processed.

**Q: Can I process multiple faces in one video?**  
A: Currently processes one face at a time (closest/largest).

**Q: What if no face is detected?**  
A: Frame is still saved with person detection (backup) if available.

**Q: How do I stop processing early?**  
A: Press 'q' - output video saved with frames processed so far.

**Q: Does it work on all video formats?**  
A: Most common formats (.mp4, .mov, .avi, .mkv) are supported.

---

## 📞 Need Help?

1. Check this guide and main `README.md`
2. Review console output for error messages
3. Try `python test_video_mode.py` for interactive testing
4. Verify file paths and permissions
5. Test with a short video clip first

---

**Last Updated:** October 2025  
**For full system documentation, see README.md**
