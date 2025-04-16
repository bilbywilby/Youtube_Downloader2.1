Here's a draft for a `README.md` file for your repository:

```markdown
# YouTube Downloader 2.1

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

## Overview

YouTube Downloader 2.1 is a powerful and user-friendly desktop application that allows you to download videos and audio from YouTube and other supported platforms. Built with `yt-dlp`, the application comes with an intuitive GUI implemented using `PySimpleGUI`, making it accessible for users of all technical levels. 

This downloader supports multiple formats, resolutions, and advanced features like metadata embedding, thumbnail extraction, and more.

---

## Features

- **Video and Audio Downloading**: Supports MP4, MKV, MP3, FLAC, and more.
- **Quality Selection**: Download videos in resolutions up to 4K.
- **Playlist Support**: Download entire playlists effortlessly.
- **GUI**: Intuitive and easy-to-use interface with progress tracking.
- **Error Handling**: Detailed logging and retry mechanisms ensure downloads are reliable.
- **Resume Capability**: Automatically resumes interrupted downloads.
- **Metadata and Thumbnail Extraction**: Save additional information for downloaded files.
- **Configurable Options**: Fully customizable settings for download preferences.

---

## Installation

### Prerequisites

Make sure you have the following installed:

- **Python 3.8 or higher**
- **FFmpeg** (Optional, for enhanced audio/video processing)

### Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/bilbywilby/Youtube_Downloader2.1.git
   cd Youtube_Downloader2.1
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python youtube_downloader.py
   ```

---

## Usage

### Steps to Download a Video or Audio:

1. Launch the application by running `youtube_downloader.py`.
2. Paste the YouTube URL into the input field.
3. Select your desired format and quality.
4. Choose whether you want to download only the audio or the entire video.
5. Click **Start Download** to begin.

### Advanced Features:

- Use the **Settings** menu to configure default download paths, themes, and more.
- Enable subtitles, playlists, and metadata extraction as needed.

---

## Configuration

The application automatically creates a configuration file (`config.json`) in the `.youtube_downloader` directory. You can update the configuration directly in the app or by editing the file.

---

## Troubleshooting

- **Insufficient Disk Space**: Ensure at least 1GB of free space is available.
- **FFmpeg Not Found**: Install FFmpeg to enable advanced features like audio extraction.
- **Invalid URL**: Confirm the URL is a valid YouTube link.

Check the log file (`~/.youtube_downloader/logs/downloader.log`) for detailed error messages.

---

## Contributing

Contributions are welcome! If you'd like to improve this project, please:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - The core library used for downloading videos.
- [PySimpleGUI](https://github.com/PySimpleGUI/PySimpleGUI) - The GUI framework.
- [FFmpeg](https://ffmpeg.org/) - For audio/video processing.

---

## Screenshots

![Main Interface](https://via.placeholder.com/800x500?text=Main+Interface)
*Example of the application's main interface.*

---

## Contact

For questions or feedback, please open an issue in the [GitHub repository](https://github.com/bilbywilby/Youtube_Downloader2.1/issues).
```

You may want to adjust the content (e.g., screenshots, additional features) to better reflect the specifics of your application.
