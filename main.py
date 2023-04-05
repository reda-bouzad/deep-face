import os
from deepface import DeepFace

# load the Facenet model for face recognition
model = DeepFace.build_model('Facenet')

# check if two photos are simular :
verify_result = DeepFace.verify(img1_path="male1.jpg", img2_path="female1.jpg")

# find the person in a photo :
find_result = DeepFace.find(img_path="Hakimi.png", db_path="/home/reda/Pictures/persons", model_name='Facenet',  enforce_detection=False)

# return the first element of the array which is the dataframe :
df = find_result[0]

# transform the dataFrame to a String
df_string = df.to_string()

df['Person'] = df['Person'] = [os.path.basename(os.path.dirname(file_path)) for file_path in df['identity']]


print('----------------------------------------------------------------------------')
max_value = df['Facenet_cosine'].max()
max_person = df.loc[df['Facenet_cosine'].idxmax(), 'Person']
print(f"person identified , his name is : {max_person}" )
print('----------------------------------------------------------------------------')

# save the data to the file dataframe.txt
with open('dataframe.txt', 'w') as file:
    file.write(df_string)

DeepFace.stream("/home/reda/Pictures/persons" , model_name="Facenet")

