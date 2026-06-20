import sounddevice as sd
import numpy as np
import winsound
import time
import queue
import argparse
import soundfile as sf

# --- Configuration ---
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000  # Process 1/4 second chunks at a time
CALIBRATION_DURATION = 3.0 # Seconds to calibrate background noise

# The spike multiplier: How many times louder than the background noise does a sound need to be?
SENSITIVITY_MULTIPLIER = 5.0 

# Minimum absolute volume threshold to prevent triggering on tiny noises in completely silent rooms
# (Range typically 0.0 to 1.0, where 1.0 is max digital volume)
MIN_ABSOLUTE_THRESHOLD = 0.05 

class AbnormalSoundDetector:
    def __init__(self):
        self.q = queue.Queue()
        self.is_calibrating = True
        self.calibration_samples = []
        self.baseline_rms = 0.0
        self.start_time = time.time()
        
    def audio_callback(self, indata, frames, time_info, status):
        """This is called continuously by sounddevice for each audio block."""
        if status:
            print(status)
        # Calculate RMS (Root Mean Square) volume for the current audio block
        rms = np.sqrt(np.mean(indata**2))
        self.q.put(rms)

    def trigger_alarm(self):
        print("\n" + "!"*50)
        print("🚨 ABNORMAL SOUND DETECTED! 🚨")
        print("!"*50 + "\n")
        
        # Play a very loud, fast, and jarring siren pattern
        for _ in range(8):
            winsound.Beep(3500, 100) # Very high piercing pitch
            winsound.Beep(1500, 100) # Sharp drop
            winsound.Beep(4500, 100) # Even higher piercing pitch
            winsound.Beep(1500, 100) # Sharp drop
            
        print("Resuming monitoring...\n")
        # Briefly pause to let echoes die down before resuming
        time.sleep(1)
        
    def process_rms(self, current_rms):
        if self.is_calibrating:
            self.calibration_samples.append(current_rms)
            
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= CALIBRATION_DURATION:
                self.is_calibrating = False
                # Calculate the average background noise level
                self.baseline_rms = np.mean(self.calibration_samples)
                # Ensure baseline isn't completely zero to avoid division by zero later
                self.baseline_rms = max(self.baseline_rms, 0.001) 
                
                print("\n--- Calibration Complete ---")
                print(f"Background Noise Level: {self.baseline_rms:.5f}")
                print(f"Alarm will trigger if volume exceeds: {max(self.baseline_rms * SENSITIVITY_MULTIPLIER, MIN_ABSOLUTE_THRESHOLD):.5f}")
                print("----------------------------\n")
                print("🟢 SYSTEM ARMED: Listening for abnormal sounds...")
                
        else:
            # ACTIVE MONITORING MODE
            # Print a visual meter of the current sound level
            meter_length = min(int(current_rms * 100), 20)
            print(f"Listening... [{'|'*meter_length:<20}] {current_rms:.4f}", end='\r')
            
            # Check if the sound is a sudden massive spike AND loud enough in general
            if current_rms > (self.baseline_rms * SENSITIVITY_MULTIPLIER) and current_rms > MIN_ABSOLUTE_THRESHOLD:
                # Alarm triggered!
                print("\n\nSpike Detected:", current_rms)
                self.trigger_alarm()
                # Clear out any queued audio blocks during the alarm so we don't double-trigger
                with self.q.mutex:
                    self.q.queue.clear()

    def start_monitoring(self, audio_file=None):
        print("Initializing Abnormal Sound Detector...")
        self.start_time = time.time()
        
        try:
            if audio_file:
                if audio_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    import os
                    import tempfile
                    from moviepy import AudioFileClip
                    print(f"Extracting audio from video file: {audio_file}...")
                    audio = AudioFileClip(audio_file)
                    temp_wav = os.path.join(tempfile.gettempdir(), "extracted_audio.wav")
                    audio.write_audiofile(temp_wav, fps=SAMPLE_RATE, logger=None)
                    audio_file = temp_wav
                    print("Audio extraction complete.")

                print(f"Reading from audio file: {audio_file}")
                # Read audio file using soundfile
                data, fs = sf.read(audio_file)
                if len(data.shape) > 1:
                    data = data[:, 0]  # Use only the first channel
                
                # Calculate block size relative to the file's sample rate
                actual_block_size = int((BLOCK_SIZE / SAMPLE_RATE) * fs)
                num_blocks = len(data) // actual_block_size
                
                print(f"CALIBRATING for {CALIBRATION_DURATION} seconds...")
                for i in range(num_blocks):
                    block_start = i * actual_block_size
                    block_end = (i + 1) * actual_block_size
                    block = data[block_start:block_end]
                    
                    rms = np.sqrt(np.mean(block**2))
                    self.process_rms(rms)
                    
                    # Simulate real-time playback speed
                    time.sleep(actual_block_size / fs)
                    
            else:
                print("Microphone starting...")
                # Start streaming audio from the default microphone
                with sd.InputStream(callback=self.audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE):
                    print(f"CALIBRATING for {CALIBRATION_DURATION} seconds. Please keep the room quiet...")
                    
                    while True:
                        current_rms = self.q.get()
                        self.process_rms(current_rms)
                        
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abnormal Sound Detector")
    parser.add_argument('--audio', type=str, default=None, help='Path to audio file (.wav, .flac) for testing instead of mic')
    args = parser.parse_args()

    detector = AbnormalSoundDetector()
    detector.start_monitoring(audio_file=args.audio)
