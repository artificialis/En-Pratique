#
#
#


import warnings

from torch import device

warnings.filterwarnings("ignore", category=UserWarning)

# Imports
import argparse
import torch
import numpy as np
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="HF_TOKEN_REDACTED"
    )
    pipeline.to(torch.device("cuda"))

    # Detect authors
    diarization_data = pipeline(args.audio_file)

    # For each entry
    for turn, _, speaker in diarization_data.itertracks(yield_label=True):
        console.print(f"start={turn.start:.3f} stop={turn.end:.3f} Speaker={int(speaker)}")
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

