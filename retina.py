from retinaface import RetinaFace

resp = RetinaFace.detect_faces("messi-neymar.jpg")
person_count = len(resp)

print(person_count)
