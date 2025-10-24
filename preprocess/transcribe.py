# Json commentary comes from another source
# This file takes care of direct transcription of audio

# Whisper needs ffmpeg installed

import argparse
import os
import sys
import json
import datetime
import whisper

def format_ts(seconds: float) -> str:
    """Format seconds as H:MM:SS.mmm (e.g., 0:01:23.456)."""
    if seconds is None:
        return "N/A"
    ms = int(round((seconds - int(seconds)) * 1000))
    td = datetime.timedelta(seconds=int(seconds))
    return f"{td}.{ms:03d}"

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe an MKV (or any audio/video) file with Whisper and print timestamps."
    )
    parser.add_argument("input", help="Path to the .mkv (or other) media file")
    parser.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (larger = better but slower). Default: small",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="ISO language code (e.g., 'en'). If omitted, Whisper will auto-detect."
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Force device. By default Whisper picks automatically."
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save raw JSON output (segments, etc.)."
    )
    parser.add_argument(
        "--srt",
        action="store_true",
        help="Also write an .srt file next to the input."
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Whisper model: {args.model} ...")
    model = whisper.load_model(args.model, device=args.device if args.device else None)

    print("Transcribing... (this may take a while depending on model size and file length)")
    result = model.transcribe(args.input, language=args.language, verbose=False, fp16=False)

    # print each segment with start and end timestamps
    segments = result.get("segments", [])
    if not segments:
        print("No segments found.")
        sys.exit(0)

    print("\n--- Transcript (with timestamps) ---\n")
    for seg in segments:
        start = format_ts(seg.get("start"))
        end = format_ts(seg.get("end"))
        text = seg.get("text", "").strip()
        print(f"[{start} --> {end}] {text}")

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON to: {args.save_json}")

    if args.srt:
        base, _ = os.path.splitext(args.input)
        srt_path = base + ".srt"
        with open(srt_path, "w", encoding="utf-8") as srt:
            for i, seg in enumerate(segments, start=1):
                start = seg.get("start", 0.0)
                end = seg.get("end", 0.0)

                def srt_ts(t):
                    h, rem = divmod(int(t), 3600)
                    m, s = divmod(rem, 60)
                    ms = int(round((t - int(t)) * 1000))
                    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

                text = seg.get("text", "").strip()
                srt.write(f"{i}\n{srt_ts(start)} --> {srt_ts(end)}\n{text}\n\n")
        print(f"Saved SRT to: {srt_path}")

if __name__ == "__main__":
    main()
