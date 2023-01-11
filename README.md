# AutomatedVAR
Yolov5 Folder: https://drive.google.com/drive/folders/1OGVun78Lc8LhvwaS-ValoH8B0v_Apx-o?usp=sharing
YouTube video of our presentation: https://www.youtube.com/watch?v=HZvvLW67gfk&t=13s

## What is this project and why does Automated VAR matter?

## What is the novelty in our project?

  For many testcases, we have automated more of the VAR process than ever before:
    - Our input is an mp4 file rather than just a jpeg, meaning that our program needs to also find the specific frame in the video to analyze for offsides
    - Our input only uses one broadcast view as opposed to other models that required the use of multiple broadcast views
    - From the input of the mp4 file to the output of the decision, every part of our process is automated except selecting which player is being passed to

## So how did we do this?

### Step 1: Player tracking and ball tracking

### Step 2: Seperating Players by team

### Step 3: What two players are in contention for offsides?


### Step 4: Identifying the Goal-line

#### Why did we do this?
- The offsides lines shown in the image above are parallel to the goal-line. So to make these offsides lines for the two players

#### What did we try?

#### What did we end up actually doing?

### Step 5: Drawing the Lines


### Future Work/Limitations
- Creating a frontend for users to enter testcases and give feedback on our model
- Sacrificing our novelty of using only one broadcast view, because using multiple broadcast views means that we could use 3D coordinates to map the positions of the players and the ball as opposed to 2D coordinates, which would lead to more accuracy. 
- Our model requires that the goal-line needs to be clearly in the frame of the video, or else we can't draw the offside lines parallel to the players.






