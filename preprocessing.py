from pdfminer.high_level import extract_text
from io import StringIO
from pdfminer.high_level import extract_text_to_fp
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from bs4 import BeautifulSoup
from pdfminer.layout import LAParams

output_string = StringIO()
with open('input.pdf', 'rb') as fin:
    extract_text_to_fp(fin, output_string, laparams=LAParams(), output_type='html', codec=None)

contents = output_string.getvalue()

with open('output_text.html', mode='w', encoding='utf-8') as f:
    print(contents, file=f)

with open('output_text.html', encoding='utf-8') as fp:
    soup = BeautifulSoup(fp, 'html.parser')

text = (soup.get_text()).strip()
lines_with_empty_line = text.split("\n")
lines = list(filter(None, lines_with_empty_line))

with open('output_lines.txt', mode='w', encoding='utf-8') as f:
    print('\n'.join(lines), file=f)

layer = preprocessing.TextVectorization()
layer.adapt(lines)
vectorized_text = layer(lines)

save_vectorized_text_file = "output_vectorized_text.txt"
print('Here is the vectorized text after preprocessing (result is saved to {}):'.format(save_vectorized_text_file))
print(vectorized_text)

with open(save_vectorized_text_file, mode='w', encoding='utf-8') as f:
    for row in range(vectorized_text.shape[0]):
        for col in range(vectorized_text.shape[1]):
            f.write('{:05} '.format(vectorized_text[row][col]))
        f.write('\n')



