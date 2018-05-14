# Subtractor
  * This is homework3 for DSAI, it can support **3-digits addition and subtraction**
# Idea
  * I've modified the TA's example RNN model to support subtraction, but the accuracy is at about 7X %. 
  * So I do some little change on the encodeing of the data and build a model which consist of several fully-connected layers (Dense layers).
  * I seperate input expression into sign and numeric, and further train them indvidually.
  * See more details in `Subtractor.ipynb`
  
# Run
  * Use exist model to perform addition/subtraction. 
    - `$ python3 main.py`
  * Retrain the model by random data
    - `$ python3 main.py --retrain=True`

# Requiements
  * `pip install -r requirements.txt`
  * keras==2.1.6
  * numpy==1.14.1

# References
  * [TA's example code](https://github.com/IKMLab/Adder-practice)
  * [keras dense layer](https://keras.io/getting-started/sequential-model-guide/)
  * [checkpoint](https://machinelearningmastery.com/check-point-deep-learning-models-keras/)
  
  
