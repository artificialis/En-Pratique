
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Imports
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
install(show_locals=True)


# Load pre-trained speaker diarization
console.print("[bold blue]Loading pre-trained speaker embedding...[/bold blue]")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="HF_TOKEN_REDACTED"
)
pipeline.to(torch.device("cuda"))

# Load the pre-trained speaker embedding model
console.print("[bold green]Loading pretrained speaker embedding model...[/bold green]")
model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda")
)

audio = Audio(mono="downmix")

# Audio files
audio_files = [
    "audio/speaker_02.wav",
    "audio/speaker_03.wav",
    "audio/speaker_04.wav",
]

# Segment audio files
console.print(f"[bold magenta]Diarizing {len(audio_files)} audio files...[/bold magenta]")
speaker_diarization = [pipeline(audio_file) for audio_file in audio_files]

# List of embeddings, speaker and files
embeddings = list()
embeddings_files = list()
embeddings_speaker = list()

# Get embeddings for each segment
for f_i, (file_name, dia) in enumerate(zip(audio_files, speaker_diarization)):
    file_duration = audio.get_duration(file_name)
    console.print(f"[bold cyan]Processing {file_name}...[/bold cyan]")
    console.print(f"[bold cyan]File duration:[/bold cyan] {file_duration}")
    for turn, _, speaker in dia.itertracks(yield_label=True):
        # Check end
        seg_end = turn.end if turn.end <= file_duration else file_duration

        # extract embedding for a speaker speaking
        speaker_segment = Segment(turn.start, seg_end)
        waveform, sample_rate = audio.crop(file_name, speaker_segment)
        speaker_embedding = model(waveform[None])

        # Add to embeddings
        embeddings.append(speaker_embedding)

        # Add to info
        embeddings_files.append(f_i)
        embeddings_speaker.append(speaker)
    # end for
# end for

# Concat embeddings
embeddings = np.concatenate(embeddings, axis=0)
console.print(f"embeddings.shape: {embeddings.shape}")

# Application de t-SNE
console.print("[bold yellow]Applying TSNE dimensionality reduction...[/bold yellow]")
tsne = TSNE(n_components=3, perplexity=30, random_state=42)
embeddings_3d = tsne.fit_transform(embeddings)

# Affichage
markers = {
    "SPEAKER_00": 'o',
    "SPEAKER_01": 'x',
    "SPEAKER_02": '^',
    "SPEAKER_03": 's',
}
emb_markers = [markers[s] for s in embeddings_speaker]

# Create 3D figure and axes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 3D scatter plot
scatter = ax.scatter(
    xs=embeddings_3d[:, 0],
    ys=embeddings_3d[:, 1],
    zs=embeddings_3d[:, 2],
    # marker=emb_markers,
    c=embeddings_files,
    cmap='tab10'
)

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Projection t-SNE 3D des embeddings")

# Set initial view angle
ax.view_init(elev=30, azim=45)

plt.show()

