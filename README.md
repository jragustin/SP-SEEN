# SP-SEEN
SEEN: Comparative study of Steganalysis using CNN, StegDetect and Zhang Ping Steganalysis

# SET-UP
## Download the repository
```
git clone https://github.com/jragustin/SP-SEEN.git
cd SP-SEEN
```
you can also download the already trained models using this link and save it to SP-SEEN/models folder

https://drive.google.com/file/d/1NdvZAGCGPYcH9-1qvBXvdkGSV4oZ7uEK/view?usp=sharing

Download all the required python modules using the ff commands:
```
sudo apt install python3-pip
pip3 install keras numpy tensorflow pillow tkinter matplotlib
```
  
## Run and Train the Neural Network
```
python3 network.py
```
you might wait for hours in order to finish the training.

## Run the classifier using this command

```
python3 ui.py
```
