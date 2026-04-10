"""
Download sample videos for the StreamMind demo and evaluation.

Videos cover diverse domains so LiveQA-Bench questions span a wide range
of visual scenarios.  All source clips are free-licensed from Mixkit.

Demo clips (3):
  cooking.mp4, surveillance.mp4, activity.mp4

Evaluation clips (28 additional, downloaded with --eval):
  Daily-life/indoor: gym, grocery, classroom, warehouse, cafe,
    workshop, restaurant, library
  Outdoor/urban: traffic, street, park_jog, night_city, market, aerial_city
  Sports/action: soccer, basketball, swimming, running, cycling, boxing
  Trailer-style/cinematic: car_chase, fireworks, concert, sunset
  Nature/wildlife: beach, forest, underwater, rain

Usage:
  python download_samples.py              # demo clips only
  python download_samples.py --eval       # demo + evaluation clips

Requires ffmpeg on PATH for building composite videos.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import urllib.request
import shutil

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)

# ── Demo videos (always downloaded) ──────────────────────────────────────

VIDEOS = {
    "cooking.mp4": {
        "url": "https://assets.mixkit.co/videos/43059/43059-720.mp4",
        "description": "Cook preparing food in a kitchen pan",
    },
    "surveillance.mp4": {
        "url": "https://assets.mixkit.co/videos/15478/15478-720.mp4",
        "description": "Empty office space, no people",
    },
}

ACTIVITY_SCENES = [
    {
        "id": "scene_office",
        "url": "https://assets.mixkit.co/videos/25591/25591-720.mp4",
        "description": "Man doing different activities at home",
    },
    {
        "id": "scene_dog",
        "url": "https://assets.mixkit.co/videos/1211/1211-720.mp4",
        "description": "Dog playing in slow motion",
    },
    {
        "id": "scene_park",
        "url": "https://assets.mixkit.co/videos/4498/4498-720.mp4",
        "description": "Woman walking in a city park",
    },
    {
        "id": "scene_cooking",
        "url": "https://assets.mixkit.co/videos/43059/43059-720.mp4",
        "description": "Cook preparing food in a kitchen",
    },
]

# ── Evaluation videos (downloaded with --eval) ──────────────────────────
# Each covers a distinct visual domain to improve benchmark diversity.

EVAL_VIDEOS = {
    # ── Daily life & indoor ───────────────────────────────────────────
    "gym.mp4": {
        "url": "https://assets.mixkit.co/videos/34563/34563-720.mp4",
        "description": "Person exercising at a gym with equipment",
    },
    "grocery.mp4": {
        "url": "https://assets.mixkit.co/videos/34588/34588-720.mp4",
        "description": "Person shopping for produce in a grocery store",
    },
    "classroom.mp4": {
        "url": "https://assets.mixkit.co/videos/4881/4881-720.mp4",
        "description": "Students and teacher in a classroom setting",
    },
    "warehouse.mp4": {
        "url": "https://assets.mixkit.co/videos/21730/21730-720.mp4",
        "description": "Interior of a large warehouse with shelving",
    },
    "cafe.mp4": {
        "url": "https://assets.mixkit.co/videos/4819/4819-720.mp4",
        "description": "Barista preparing coffee in a cafe",
    },
    "workshop.mp4": {
        "url": "https://assets.mixkit.co/videos/4866/4866-720.mp4",
        "description": "Craftsperson working with tools in a workshop",
    },
    "restaurant.mp4": {
        "url": "https://assets.mixkit.co/videos/3296/3296-720.mp4",
        "description": "Waitstaff serving diners in a restaurant",
    },
    "library.mp4": {
        "url": "https://assets.mixkit.co/videos/4858/4858-720.mp4",
        "description": "People reading and studying in a library",
    },

    # ── Outdoor & urban ───────────────────────────────────────────────
    "traffic.mp4": {
        "url": "https://assets.mixkit.co/videos/3888/3888-720.mp4",
        "description": "Cars driving on a busy highway at daytime",
    },
    "street.mp4": {
        "url": "https://assets.mixkit.co/videos/4397/4397-720.mp4",
        "description": "Pedestrians walking on a busy city sidewalk",
    },
    "park_jog.mp4": {
        "url": "https://assets.mixkit.co/videos/2321/2321-720.mp4",
        "description": "Person jogging through a park with trees",
    },
    "night_city.mp4": {
        "url": "https://assets.mixkit.co/videos/4064/4064-720.mp4",
        "description": "City skyline and lights at night, urban timelapse",
    },
    "market.mp4": {
        "url": "https://assets.mixkit.co/videos/2899/2899-720.mp4",
        "description": "Busy outdoor market with stalls and shoppers",
    },
    "aerial_city.mp4": {
        "url": "https://assets.mixkit.co/videos/3561/3561-720.mp4",
        "description": "Aerial drone view of a city with buildings",
    },

    # ── Sports & action ───────────────────────────────────────────────
    "soccer.mp4": {
        "url": "https://assets.mixkit.co/videos/2309/2309-720.mp4",
        "description": "Soccer players passing and shooting on a field",
    },
    "basketball.mp4": {
        "url": "https://assets.mixkit.co/videos/499/499-720.mp4",
        "description": "Basketball players on court during a game",
    },
    "swimming.mp4": {
        "url": "https://assets.mixkit.co/videos/4277/4277-720.mp4",
        "description": "Swimmer doing laps in a pool, close-up strokes",
    },
    "running.mp4": {
        "url": "https://assets.mixkit.co/videos/587/587-720.mp4",
        "description": "Athletes sprinting on a running track",
    },
    "cycling.mp4": {
        "url": "https://assets.mixkit.co/videos/2285/2285-720.mp4",
        "description": "Cyclist riding fast on a road or trail",
    },
    "boxing.mp4": {
        "url": "https://assets.mixkit.co/videos/12759/12759-720.mp4",
        "description": "Boxer training with punching bag in a gym",
    },

    # ── Trailer-style / cinematic ─────────────────────────────────────
    "car_chase.mp4": {
        "url": "https://assets.mixkit.co/videos/2538/2538-720.mp4",
        "description": "Fast car driving on a highway, cinematic shot",
    },
    "fireworks.mp4": {
        "url": "https://assets.mixkit.co/videos/547/547-720.mp4",
        "description": "Fireworks display exploding over night sky",
    },
    "concert.mp4": {
        "url": "https://assets.mixkit.co/videos/3478/3478-720.mp4",
        "description": "Concert crowd with stage lights and performer",
    },
    "sunset.mp4": {
        "url": "https://assets.mixkit.co/videos/4779/4779-720.mp4",
        "description": "Dramatic sunset over landscape with clouds",
    },

    # ── Nature & wildlife ─────────────────────────────────────────────
    "beach.mp4": {
        "url": "https://assets.mixkit.co/videos/1227/1227-720.mp4",
        "description": "Waves crashing on a sandy beach, coastal view",
    },
    "forest.mp4": {
        "url": "https://assets.mixkit.co/videos/3379/3379-720.mp4",
        "description": "Dense forest canopy, aerial view through trees",
    },
    "underwater.mp4": {
        "url": "https://assets.mixkit.co/videos/3269/3269-720.mp4",
        "description": "Underwater coral reef with tropical fish",
    },
    "rain.mp4": {
        "url": "https://assets.mixkit.co/videos/2543/2543-720.mp4",
        "description": "Heavy rain falling on a street, puddles forming",
    },
}


def download_file(url, dest, description=""):
    """Download a single file. Returns True on success."""
    if os.path.exists(dest):
        print(f"  [skip] {os.path.basename(dest)} already exists")
        return True

    print(f"  Downloading {os.path.basename(dest)} ...")
    if description:
        print(f"    {description}")
    print(f"    URL: {url}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "StreamMind-Demo/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"    Saved ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"    ERROR: {e}")
        print(f"    You can manually download from: {url}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def has_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def build_activity_video():
    """Download 4 distinct scenes and concatenate them into activity.mp4."""
    dest = os.path.join(SAMPLE_DIR, "activity.mp4")
    if os.path.exists(dest):
        print(f"  [skip] activity.mp4 already exists")
        return

    if not has_ffmpeg():
        print("\n  WARNING: ffmpeg not found on PATH.")
        print("  Cannot build activity.mp4 (multi-scene composite).")
        print("  Install ffmpeg:  brew install ffmpeg  (macOS)")
        print("                   apt install ffmpeg   (Linux)")
        print("  Then re-run this script.\n")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        scene_paths = []
        for scene in ACTIVITY_SCENES:
            scene_file = os.path.join(tmpdir, f"{scene['id']}.mp4")
            ok = download_file(scene["url"], scene_file, scene["description"])
            if not ok:
                print(f"  Skipping activity.mp4 build (missing scene: {scene['id']})")
                return
            scene_paths.append(scene_file)

        normalized = []
        for i, path in enumerate(scene_paths):
            out = os.path.join(tmpdir, f"norm_{i}.mp4")
            subprocess.run([
                "ffmpeg", "-y", "-i", path,
                "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,"
                       "pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24",
                "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-t", "20",
                out,
            ], capture_output=True, check=True)
            normalized.append(out)
            print(f"    Normalized scene {i+1}/{len(scene_paths)}")

        concat_file = os.path.join(tmpdir, "concat.txt")
        with open(concat_file, "w") as f:
            for p in normalized:
                f.write(f"file '{p}'\n")

        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c", "copy", dest,
        ], capture_output=True, check=True)

        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  Built activity.mp4 ({size_mb:.1f} MB, 4 scenes × ~20s)")


def main():
    parser = argparse.ArgumentParser(description="Download StreamMind sample videos")
    parser.add_argument("--eval", action="store_true",
                        help="Also download evaluation-only videos (28 extra clips)")
    args = parser.parse_args()

    print("StreamMind — downloading sample videos\n")
    print(f"Output directory: {os.path.abspath(SAMPLE_DIR)}\n")

    print("--- Demo clips ---")
    for name, info in VIDEOS.items():
        download_file(info["url"], os.path.join(SAMPLE_DIR, name), info["description"])

    print("\n--- Multi-scene composite (recommended for demo) ---")
    build_activity_video()

    if args.eval:
        print("\n--- Evaluation clips (28 diverse domains) ---")
        for name, info in EVAL_VIDEOS.items():
            download_file(info["url"], os.path.join(SAMPLE_DIR, name), info["description"])

    n_files = len([f for f in os.listdir(SAMPLE_DIR) if f.endswith(".mp4")])
    print(f"\nDone. {n_files} videos in {os.path.abspath(SAMPLE_DIR)}")


if __name__ == "__main__":
    main()
