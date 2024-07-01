import requests
from PIL import Image

url = """https://media.geeksforgeeks.org/\ 
wp-content/uploads/20210224040124/JSBinColla\ 
borativeJavaScriptDebugging6-300x160.png"""

# This statement requests the resource at
# the given link, extracts its contents
# and saves it in a variable
data = requests.get(url).content

# Opening a new file named img with extension .jpg
# This file would store the data of the image file
f = open('img.jpg', 'wb')

# Storing the image data inside the data variable to the file
f.write(data)
f.close()

# Opening the saved image and displaying it
img = Image.open('img.jpg')
img.show()
