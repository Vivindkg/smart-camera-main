import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Smart Camera System - Unified Entry Point")
    subparsers = parser.add_subparsers(dest="monitor", help="Available Monitors")

    # Adaptive Stream
    parser_adaptive = subparsers.add_parser("adaptive", help="Adaptive Stream Monitor")
    parser_adaptive.add_argument('--video', type=str, default='0', help='Path to video file or camera index')
    parser_adaptive.add_argument('--output', type=str, default='adaptive_output.mp4', help='Path to save output video')

    # ALPR
    parser_alpr = subparsers.add_parser("alpr", help="Automatic License Plate Recognition")
    parser_alpr.add_argument('--video', type=str, default='0', help="Path to video file or '0' for webcam")
    parser_alpr.add_argument('--log', type=str, default='license_plates.log', help="File to save the read plates")

    # Queue Monitor
    parser_queue = subparsers.add_parser("queue", help="Queue Monitor")
    parser_queue.add_argument('--video', type=str, default='0', help='Path to video file or camera index')
    parser_queue.add_argument('--output', type=str, default='queue_output.mp4', help='Path to save output video')

    # Traffic Monitor
    parser_traffic = subparsers.add_parser("traffic", help="Traffic Monitor")
    parser_traffic.add_argument('--video', type=str, default='0', help='Path to video file or camera index')
    parser_traffic.add_argument('--output', type=str, default='traffic_output.mp4', help='Path to save output video')

    # People Counter
    parser_people = subparsers.add_parser("people", help="People Counter")
    parser_people.add_argument('--video', type=str, default='0', help='Path to video file or camera index')
    parser_people.add_argument('--output', type=str, default='output_counted_video.mp4', help='Path to save output video')

    # Facial Recognition
    parser_face = subparsers.add_parser("face", help="Facial Recognition")
    parser_face.add_argument('--video', type=str, default='sample_traffic.mp4', help='Path to the input video file or camera index (e.g. 0)')
    parser_face.add_argument('--output', type=str, default='face_output.mp4', help='Path to save the output video')
    parser_face.add_argument('--known_dir', type=str, default='known_workers', help='Directory containing known faces')

    # Access Control
    parser_access = subparsers.add_parser("access", help="Access Control")
    parser_access.add_argument('--video', type=str, default='0', help='Path to the input video file or camera index (e.g. 0)')
    parser_access.add_argument('--known_dir', type=str, default='known_workers', help='Directory containing known faces')

    # Worker Tracker
    parser_worker = subparsers.add_parser("worker", help="Worker Activity Tracker")
    parser_worker.add_argument('--video', type=str, default='0', help='Path to the input video file (.mp4) or camera index (e.g. 0)')
    parser_worker.add_argument('--output', type=str, default='worker_tracker_output.mp4', help='Path to save the output video')

    # Abnormal Sound Detector
    parser_sound = subparsers.add_parser("sound", help="Abnormal Sound Detector")
    parser_sound.add_argument('--audio', type=str, default=None, help='Path to audio file (.wav, .flac) for testing instead of mic')

    # Security Monitor
    parser_security = subparsers.add_parser("security", help="Comprehensive Security Monitor")
    parser_security.add_argument('--list_audio', action='store_true', help="List all available audio input devices and exit")
    parser_security.add_argument('--audio_device', type=int, default=None, help="The ID of the external microphone to use")
    parser_security.add_argument('--output', type=str, default='security_recording.mp4', help="Path to save the final recording with sound")

    args = parser.parse_args()

    if args.monitor == "adaptive":
        from src.monitors.adaptive_stream import run
        run(args.video, args.output)
    elif args.monitor == "alpr":
        from src.monitors.alpr_monitor import run
        run(args.video, args.log)
    elif args.monitor == "queue":
        from src.monitors.queue_monitor import run
        run(args.video, args.output)
    elif args.monitor == "traffic":
        from src.monitors.traffic_monitor import run
        run(args.video, args.output)
    elif args.monitor == "people":
        from src.monitors.people_counter import run
        run(args.video, args.output)
    elif args.monitor == "face":
        from src.monitors.facial_recognition import run
        run(args.video, args.output, args.known_dir)
    elif args.monitor == "access":
        from src.monitors.access_control import run
        run(args.video, args.known_dir)
    elif args.monitor == "worker":
        from src.monitors.worker_tracker import run
        run(args.video, args.output)
    elif args.monitor == "sound":
        from src.monitors.abnormal_sound_detector import run
        run(args.audio)
    elif args.monitor == "security":
        from src.monitors.security_monitor import run
        run(args.list_audio, args.audio_device, args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
