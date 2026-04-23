import cv2
from deepface import DeepFace
from collections import Counter

MAP_TO_4 = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "disgust": "angry",
    "neutral": "neutral",
    "fear": "neutral",
    "surprise": "neutral"
}

def test_emotion(samples=5, confidence_threshold=0.5):
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print(f"Taking {samples} frames. Look at the camera now.\n")

    mapped_samples = []
    skipped = 0

    for i in range(samples):
        ret, frame = cap.read()
        if not ret:
            print(f"  Frame {i+1}: Could not read frame, skipping.")
            continue

        # Crop to face region
        h, w = frame.shape[:2]
        x1, y1 = int(w * 0.25), int(h * 0.15)
        x2, y2 = int(w * 0.75), int(h * 0.90)
        frame = frame[y1:y2, x1:x2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = DeepFace.analyze(
            rgb,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False
        )

        emotions_dict = result[0]["emotion"]
        dominant = result[0]["dominant_emotion"]
        confidence = emotions_dict[dominant] / 100.0

        # Print ALL emotion scores so you can see what DeepFace is thinking
        print(f"  Frame {i+1}:")
        print(f"    Dominant emotion : {dominant}")
        print(f"    Confidence       : {confidence:.1%}")
        print(f"    All scores:")
        for emotion, score in sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score / 5)
            print(f"      {emotion:<10} {score:5.1f}%  {bar}")

        if confidence >= confidence_threshold:
            mapped = MAP_TO_4.get(dominant.lower(), "neutral")
            mapped_samples.append(mapped)
            print(f"    → Accepted as: {mapped}\n")
        else:
            skipped += 1
            print(f"    → SKIPPED (confidence {confidence:.1%} is below {confidence_threshold:.1%} threshold)\n")

    cap.release()

    print("=" * 40)
    print(f"Frames accepted : {len(mapped_samples)}")
    print(f"Frames skipped  : {skipped}")

    if not mapped_samples:
        print("Final result    : neutral (no frames were confident enough)")
    else:
        final = Counter(mapped_samples).most_common(1)[0][0]
        print(f"Votes           : {dict(Counter(mapped_samples))}")
        print(f"Final result    : {final}")

if __name__ == "__main__":
    test_emotion()