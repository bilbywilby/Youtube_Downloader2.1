# YouTube_Downloader

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Version](https://img.shields.io/badge/Version-2.0.3-green)
![Updated](https://img.shields.io/badge/Updated-2025--04--16-brightgreen)

A powerful desktop application for downloading videos and audio from YouTube with advanced features and an intuitive GUI.

## Features

- **High-Quality Downloads**
  - Support for resolutions up to 4K
  - Multiple video formats (MP4, MKV, WEBM)
  - Various audio formats (MP3, WAV, M4A, AAC, FLAC)

- **Advanced Options**
  - Playlist support
  - Subtitle downloads
  - Thumbnail extraction
  - Metadata embedding
  - Download queue management

- **User Experience**
  - Modern graphical interface
  - Progress tracking
  - Download history
  - System tray integration
  - Keyboard shortcuts

## Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg (optional, for enhanced media processing)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/your-username/youtube-downloader-pro.git
cd youtube-downloader-pro
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python youtube_downloader.py
```

## Usage

1. **Basic Download**
   - Launch the application
   - Paste a YouTube URL
   - Select format and quality
   - Click "Start Download"

2. **Advanced Features**
   - Queue multiple downloads
   - Extract audio only
   - Download entire playlists
   - Configure custom save locations
   - Enable subtitle downloads

## Configuration

The application creates a `.youtube_downloader` directory in your home folder with:
- `config.json` - Application settings
- `history.json` - Download history
- `queue.json` - Active download queue
- `logs/` - Application logs

## Keyboard Shortcuts

- `Ctrl+Q` - Exit application
- `Ctrl+A` - Add to queue
- `Ctrl+S` - Start download
- `Ctrl+D` - Detect format
- `Ctrl+P` - Preview
- `Ctrl+C` - Cancel download
- `Ctrl+O` - Open downloads folder

## Troubleshooting

Common issues and solutions:

- **Download Fails**: Check your internet connection and URL validity
- **FFmpeg Missing**: Install FFmpeg for full format support
- **Disk Space Error**: Ensure at least 1GB free space
- **Format Unavailable**: Try a different quality or format

Check `~/.youtube_downloader/logs/downloader.log` for detailed error messages.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [PySimpleGUI](https://github.com/PySimpleGUI/PySimpleGUI)
- [FFmpeg](https://ffmpeg.org/)

## Author

[bilbywilby](https://github.com/bilbywilby)

---

For bug reports and feature requests, please [open an issue](https://github.com/your-username/youtube-downloader-pro/issues).
```
