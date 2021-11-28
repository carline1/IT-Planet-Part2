# IT-Planet-Part2

The second part of the International Olympiad IT Planet, OCR (Summer  2021)

At the second stage of the competition, it was necessary to create a model capable of recognizing the readings of water meters.

The model turned out to be unfinished, the accuracy is small.

Main.py has a function extract_image_features, which takes a image path as input and returns the following data:

<pre>
result_dict = {
    'prediction': num_pred, 
    'x1': coords[2], 
    'y1': coords[0], 
    'x2': coords[3], 
    'y2': coords[1], 
}
</pre>
