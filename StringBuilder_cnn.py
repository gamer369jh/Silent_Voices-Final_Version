import cv2
import time
import HandTraking as ht
import enchant
from gtts import gTTS
import subprocess
import HandSignDetector as hsd



def VidCap():
    s=""
    cap = cv2.VideoCapture(0)
    tracker = ht.handTracker()
    last_check_time = time.time()
    model_path = 'cnn_model.h5'  # Replace with the path to your model file
    detector = hsd.HandSignDetector(model_path)
    counter=0
    while True:
        current_time = time.time()

        success, image = cap.read()
        image = tracker.handsFinder(image)
        lmList = tracker.positionFinder(image)

        if current_time - last_check_time >= 0.5:
            last_check_time = current_time  # Update the last check time every 2 seconds
            if lmList:
                print("List is not empty\n")
                counter=0
                print(lmList,"\n")
                hand_image = tracker.getHandImage(image)
                if hand_image is not None:
                    cv2.imwrite("hand_image.png", hand_image)
                    predected_char=detector.detect_sign2("hand_image.png")
                s=s+predected_char
                # You can perform any necessary actions here if the list is not empty.
            else:
                print("List is empty\n")
                counter+=1
                if counter>=2 and counter<6:
                    s=s+" "
                if counter >=7:
                    s=s+"\n"
                    cv2.destroyAllWindows()
                    return s

        cv2.imshow("Video", image)
        key = cv2.waitKey(1)
        if key == ord('a'):
            break
    cv2.destroyAllWindows()
    s=s+"\n"
    return s


def SpellChecker(input_text):
    dictionary = enchant.Dict("en_US")
    words = input_text.split()
    corrected_words = []

    for word in words: 
        if not dictionary.check(word):
            suggestions = dictionary.suggest(word)
            if suggestions:
                corrected_words.append(suggestions[0])  # Use the first suggestion
            else:
                corrected_words.append(word)  # If no suggestions, keep the original word
        else:
            corrected_words.append(word)

    corrected_text = ' '.join(corrected_words)
    return corrected_text


def StringtoVoice(text):
    tts = gTTS(text)
    # Save the generated speech to an audio file
    tts.save("output.mp3")
    # Play the audio file (you may need to install a media player like vlc)
    subprocess.Popen(['start', "output.mp3"], shell=True)


def SignToSound():
    s=VidCap()
    corrected_text = SpellChecker(s)
    StringtoVoice(corrected_text)


def main():
    s=VidCap()
    print(s)
    corrected_text = SpellChecker(s)
    print(corrected_text,"\n")
    StringtoVoice(corrected_text)

if __name__ == "__main__":
    main()