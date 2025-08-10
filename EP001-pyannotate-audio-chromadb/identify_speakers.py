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
import numpy as np
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from rich.console import Console
from rich.traceback import install


# Configure Rich
console = Console()
install(show_locals=False)


DB_PATH = os.path.abspath("./chroma_db")


def search_speaker_embedding(
        collection_name: str,
        speaker_name: str,
        embedding: np.ndarray
):
    """
    Search for a speaker in the vector DB.

    Args:
        collection_name (str): The collection name.
        speaker_name (str): The speaker name.
        embedding (np.ndarray): The embedding.
    """
    # Create ChromaDB client
    client = chromadb.PersistentClient(path=DB_PATH)

    # Create the vector collection
    collection = client.get_collection(
        name=collection_name
    )

    # Get neighbors
    res = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=5,
        include=["distances", "metadatas", "documents"]
    )

    # Get ids and distances
    dists = res["distances"][0]
    sims = [1.0 - d for d in dists]

    console.print(f"For speaker {speaker_name}:")
    for i, (s, md, doc) in enumerate(zip(sims, res["metadatas"][0], res["documents"][0])):
        console.print(f"{i + 1}. sim={s:.4f}  meta={md}  doc={doc[:80]}")
    # end for
# end search_speaker_embedding


def main(args):
    """
    Main
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

    # Audio duration
    duration = audio.get_duration(args.audio_file)

    # Keep speaker embeddings
    speaker_embeddings = dict()

    # For each entry
    for turn, _, speaker in diarization_data.itertracks(yield_label=True):
        # Show log
        console.print(f"start={turn.start:.3f} stop={turn.end:.3f} Speaker={speaker}")

        # Segment start and end
        seg_stat = turn.start
        seg_end = turn.end if turn.end <= duration else duration

        # Get speaker embedding
        speaker_segment = Segment(seg_stat, seg_end)
        waveform, sample_rate = audio.crop(args.audio_file, speaker_segment)
        segment_embedding = model(waveform[None])

        # Add to speaker embedding
        if speaker not in speaker_embeddings:
            speaker_embeddings[speaker] = []
        # end if
        speaker_embeddings[speaker].append(segment_embedding)
    # end for

    # Concat embeddings
    speaker_embeddings = {
        name: np.concatenate(embs, axis=0)
        for name, embs in speaker_embeddings.items()
    }

    # Do average over batch dim
    speaker_embeddings = {name: np.mean(embs, axis=0) for name, embs in speaker_embeddings.items()}

    # Try to identify the speaker from the vector database
    for speaker_name, embeddings in speaker_embeddings.items():
        search_speaker_embedding(
            collection_name=args.chroma_collection,
            speaker_name=speaker_name,
            embedding=embeddings
        )
    # end for
# end main


if __name__ == "__main__":
    # Parse argument
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument("-a", "--audio-file", type=str, required=True, help="Path to the audio file")
    parser.add_argument("-c", "--chroma-collection", type=str, required=True, help="Name of the chroma collection")
    args = parser.parse_args()

    # Call main
    main(args)
# end if


