import cv2
import numpy as np

# Συνάρτηση αφαίρεσης υποβάθρου
def background_subtraction(reference_image, current_frame, threshold=50):
    difference = cv2.absdiff(reference_image, current_frame)
    _, subtracted = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)
    return subtracted

# Διαδικασία ανάγνωσης του βίντεο
video_path = r'C:\Users\eeean\Downloads\ice-timeline.mp4'


video_capture = cv2.VideoCapture(video_path)
measurements = []

# Διάβασμα του πρώτου πλαισίου ως εικόνα αναφοράς
ret, reference_frame = video_capture.read()
if not ret:
    print("Error: Unable to read video file or video file has no frames.")
    video_capture.release()
    exit()

# Μετατροπή της εικόνας αναφοράς σε κλίμακα του γκρι
reference_image = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)

# Επεξεργασία του βίντεο
while True:
    # Διάβασμα του τρέχοντος πλαισίου από το βίντεο
    ret, frame = video_capture.read()
    if not ret:
        break  # Διακοπή της επανάληψης αν δεν υπάρχουν πλαίσια

    # Μετατροπή του τρέχοντος πλαισίου σε κλίμακα του γκρι
    current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Αφαίρεση υποβάθρου
    foreground_mask = background_subtraction(reference_image, current_frame)
                                                                                                                        
    # Εμφάνιση του τρέχοντος πλαισίου με τη μάσκα
    masked_frame = cv2.bitwise_and(frame, frame, mask=foreground_mask)
    cv2.imshow('Masked Frame', masked_frame)
    cv2.waitKey(1)

    # Υπολογισμός του αριθμού μη μηδενικών στη μάσκα
    num_changes = np.count_nonzero(foreground_mask)
    measurements.append(num_changes)

    # Ανανέωση της εικόνας αναφοράς
    reference_image = current_frame

# Απελευθέρωση του αντικειμένου λήψης βίντεο και κλείσιμο όλων των πλαισίων
video_capture.release()
cv2.destroyAllWindows()

# Υπολογισμός παρεμβολής
x = [i for i in range(len(measurements))]
interpolation = np.interp(x, range(len(measurements)), measurements)

# Υπολογισμός MSE
mse = np.mean((np.array(measurements) - interpolation)**2)
print("Mean Squared Error:", mse)

# Υπολογισμός εκτιμήσεων παρεμβολής για x=500
value_at_500 = np.interp(500, range(len(measurements)), measurements)
interpolation_at_500 = np.interp(500, range(len(measurements)), interpolation)
print("Measurement at x=500:", value_at_500)
print("Interpolation at x=500:", interpolation_at_500)

# Παράθυρο γραφικών
plt.figure(figsize=(20, 10))

# Αναπαράσταση μετρήσεων και παρεμβολής
plt.subplot(1, 2, 1)
plt.plot(x, measurements, label)
