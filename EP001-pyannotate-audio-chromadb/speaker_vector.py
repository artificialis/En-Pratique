#
#
#


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Imports
import os
import argparse
import torch
import chromadb
from chromadb.config import Settings
import numpy as np
import uuid
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


DB_PATH = os.path.abspath("./chroma_db")


# Add vector to chroma DB
def add_speaker_embedding(
        collection_name: str,
        speaker_name: str,
        embedding: np.ndarray
):
    """
    Add speaker embedding to the vector database.

    Args:
        collection_name: name of the collection
        speaker_name: Name of the speaker.
        embedding: Embedding of the speaker.
    """
    # Create ChromaDB client
    client = chromadb.PersistentClient(path=DB_PATH)

    # Create the vector collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Add speaker embedding to the db
    collection.add(
        ids=[str(uuid.uuid4())],  # ID unique
        embeddings=[embedding.tolist()],  # ton vecteur
        metadatas=[{"speaker": speaker_name}],  # clé-valeur
        documents=["texte"]  # string libre
    )

    # Print
    console.print(f"Embedding for {speaker_name} was added to the collection [bold green]{collection_name}[/bold green].")
# end add_speaker_embedding


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
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}")
    pipeline.to(device)

    # Load pre-trained speaker embedding
    model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb",
        device=device
    )

    # Audio handle
    audio = Audio(mono="downmix")

    # Detect authors
    diarization_data = pipeline(args.audio_file)

    # Speaker names
    speaker_names = [args.speaker0, args.speaker1, args.speaker2]

    # Keep embeddings for each speaker
    speaker_embeddings = {name:list() for name in speaker_names}

    # Audio duration
    duration = audio.get_duration(args.audio_file)

    # For each entry
    for turn, _, speaker in diarization_data.itertracks(yield_label=True):
        # Parse speaker number
        _, speaker = speaker.split("_")

        # Speaker name
        speaker_name = speaker_names[int(speaker)]

        # Show log
        console.print(f"start={turn.start:.3f} stop={turn.end:.3f} Speaker={speaker_name}")

        # Segment start and end
        seg_stat = turn.start
        seg_end = turn.end if turn.end <= duration else duration

        # Get speaker embedding
        speaker_segment = Segment(seg_stat, seg_end)
        waveform, sample_rate = audio.crop(args.audio_file, speaker_segment)
        segment_embedding = model(waveform[None])

        # Add to speaker embedding
        speaker_embeddings[speaker_name].append(segment_embedding)
    # end for

    # Concat embeddings
    speaker_embeddings = {
        name:np.concatenate(embs, axis=0) if embs else np.empty((0, 192))
        for name, embs in speaker_embeddings.items()
    }

    # Do average over batch dim
    speaker_embeddings = {name:np.mean(embs, axis=0) for name, embs in speaker_embeddings.items()}

    # Add embeddings to the vector db
    for speaker_name, embeddings in speaker_embeddings.items():
        if speaker_name:
            add_speaker_embedding(
                collection_name=args.chroma_collection,
                speaker_name=speaker_name,
                embedding=embeddings
            )
        # end if
    # end for
# end main


if __name__ == "__main__":
    # Parse argument
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument("-a", "--audio-file", type=str, required=True, help="Path to the audio file")
    parser.add_argument("-c", "--chroma-collection", type=str, required=True, help="Name of the chroma collection")
    parser.add_argument("-s0", "--speaker0", type=str, required=False, default=None, help="Name of the speaker 0")
    parser.add_argument("-s1", "--speaker1", type=str, required=False, default=None, help="Name of the speaker 1")
    parser.add_argument("-s2", "--speaker2", type=str, required=False, default=None, help="Name of the speaker 2")
    args = parser.parse_args()

    # Call main
    main(args)
# end if

