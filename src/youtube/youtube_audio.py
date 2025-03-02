import yt_dlp
from pydub import AudioSegment

class YoutubeAudioLoader:
    def __init__(self):
        self.ydl_opts = {
                "format": "bestaudio/best",
                "extractaudio" : True,
                "quiet": True, 
                "noplaylist" : True,
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }],
                "postprocessor_args": ["-t", "180"],
                "progresshooks" : [lambda d: print(f"Status: {d['status']}")],
                "cookiesfrombrowser" : ('chrome',),
            }

    def get_mp3_from_youtube(self, url : str, filename : str = 'temp'):
        self.ydl_opts['outtmpl'] = filename
        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            print(f'File {filename} created.')

    def convert_mp3_to_wav(self, mp3file : str, output_filename : str, sr : int = 16000):
        audio = AudioSegment.from_mp3(mp3file)
        audio = audio.set_frame_rate(sr).set_channels(1).set_sample_width(2)
        audio.export(output_filename, format="wav")
        print(f"WAV file saved as: {output_filename}")