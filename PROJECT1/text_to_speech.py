from pathlib import Path
from openai import OpenAI

client = OpenAI()

def convert_text_to_speech(text: str, voice: str = "alloy", filename: str = "speech.mp3") -> Path:
    """Convert text to speech using OpenAI API and save as an MP3 file."""
    print("Starting text-to-speech conversion...")
    speech_file_path = Path(__file__).parent / filename
    try:
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=text,
        )
        with open(speech_file_path, 'wb') as speech_file:
            for chunk in response.iter_bytes():
                speech_file.write(chunk)
        print(f"Conversion completed successfully. File saved as '{filename}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return speech_file_path

if __name__ == "__main__":
    text_to_convert = "Every student in FDU can build amazing applications in minutes"
    for voice in ["alloy", "ash", "coral", "echo", "fable"]:
        filename = f"speech_{voice}hd.mp3"
        convert_text_to_speech(text_to_convert, voice, filename)