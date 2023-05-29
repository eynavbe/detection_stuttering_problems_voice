# detection stuttering problems voice


The project is performed by using the Wav2vec model to convert audio to a matrix representation, and Agnostic BERT to create vector representations from the transcripts. These represent sentences are concatenated to form a combined vector. A binary classifier then uses the data to distinguish between "stutterers" and "non-stutterers". If he stuttered, the percentage of stuttering, the signs of stuttering were obtained and the seconds during the audio in which stuttering was detected.
<br>
The main goal of this project is to propose a different approach that combines audio and text representations for the classification of stuttering instances and to help people with stuttering to get feedback on their inaccuracy, practice their speech and improve it.


- Attached is the article that expands on the project.


## User side: training and improving speech
A person who experiences speaking with a stutter will be able to practice speaking and improve it with the help of the feedback he will receive. With the help of a platform the user will fill in the details of the target text he wants to say and record himself for 5 seconds or alternatively upload a wav file of the recording with the stuttered speech. By clicking send, the information will be sent for analysis. the user will receive feedback. The feedback received is the seconds during the time of the audio where stuttering was found, the percentage of stuttering detected, the signs that the speaker had a stuttering problem.
<br><br>
<img width="602" alt="1" src="https://github.com/eynavbe/detection_stuttering_problems_voice/assets/93534494/bc959625-e3eb-41c5-a783-5367a6b453b0">

### The speaker stutters
If stuttering is detected in the recording, the user will receive some details that will help him improve himself and see progress.
<br><br>
<img width="605" alt="2" src="https://github.com/eynavbe/detection_stuttering_problems_voice/assets/93534494/99fd3d4f-6a6e-44b2-9fbb-fc41312b4b05">


### The speaker no stutters
When audio is not detected as stuttering then it will only show that it is not stuttering.
<br><br>
<img width="607" alt="3" src="https://github.com/eynavbe/detection_stuttering_problems_voice/assets/93534494/c3022997-8193-4a40-9de8-28d7555b88ce">

