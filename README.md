# self-steering-scooter
Project video: https://www.youtube.com/watch?v=sbeGKYIU3LI

HOW TO RUN:
- Only use the balance_only, go_foward, and go_anywhere folders. The rest are work in progress. 
- In each folder, run python replay.py to generate the animations used in the project video. The output is usally test.gif or test.mp4. You might need to install a few dependencies. 
- To retrain the network, run python baseline.py in each folder. During training, progress will be logged into tensorboard, and renderings will be generated once in a while.
