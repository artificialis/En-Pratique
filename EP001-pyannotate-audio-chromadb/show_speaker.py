#
#
#


import warnings

from torch import device

warnings.filterwarnings("ignore", category=UserWarning)

# Imports
import argparse
import json
import os
import torch
from pyannote.audio import Pipeline
from rich.console import Console
from rich.traceback import install

# Configure Rich
console = Console()
install(show_locals=False)


def main(args):
    """
    Main method

    Args:
        args (argparse.Namespace): arguments from argparse
    """
    # Create pipeline
    console.print("Loading pre-trained speaker diarization model...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="HF_TOKEN_REDACTED"
    )
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}")
    pipeline.to(device)

    # Load speaker mapping if available
    speaker_mapping = {}
    speakers_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speakers.json")
    
    if os.path.exists(speakers_json_path):
        try:
            with open(speakers_json_path, 'r') as f:
                speaker_data = json.load(f)
            
            # Get the base filename from the audio file path
            audio_filename = os.path.basename(args.audio_file)
            
            # Check if this file has speaker mappings
            if audio_filename in speaker_data:
                speaker_mapping = speaker_data[audio_filename]
                console.print(f"Loaded speaker mapping for {audio_filename}")
            else:
                console.print(f"No speaker mapping found for {audio_filename}")
        except Exception as e:
            console.print(f"[bold red]Error loading speakers.json: {str(e)}[/bold red]")
    else:
        console.print("speakers.json not found, using generic speaker IDs")

    # Detect authors
    console.print(f"Diarization of file {args.audio_file}")
    diarization_data = pipeline(args.audio_file)

    # For each entry
    for turn, _, speaker in diarization_data.itertracks(yield_label=True):
        # Get human-readable name if available
        human_readable_name = speaker_mapping.get(speaker, speaker)
        console.print(f"start={turn.start:.3f} stop={turn.end:.3f} Speaker={human_readable_name} ({speaker})")
    # end for
# end main


if __name__ == "__main__":
    # Parse argument
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument("-a", "--audio-file", type=str, required=True, help="Path to the audio file")
    args = parser.parse_args()

    # Call main
    main(args)
# end if

