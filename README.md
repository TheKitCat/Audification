# Audification

Audification of multidimensional Data with Python. Students project for 
Design of Multimediasystems in HS Fulda MSc. Applied Computersience program.


#How it works

The program tranforms multidimensional numeric (this ist important!) data, from a csv file, 
into 2D format through the 
following methods:


-PCA

-Random Projection

-t-SNE.


You can choose between the methods above. Afterwards a audification is made through a .wav file.



#Run the program


Start the Python script with the following arguments:

python audification [args]

-m --method : The transformation method [pca, random, tsne] : Required
-r --rounds : How many audification rounds should be taken : Default 1
-n --numThreads: Number of threads for audification : Default 1
-fr --fileWriting : Whether an audio file should be written or not : Default false

#What will be the Result?
You will get some measures, how long the whole audification lasts.
You will get some measures, how long a single transformation for each flow lasts.
You will find some interesting plots of the data in the repository.
There will be audio files for each transformation methods. Have fun :-)
