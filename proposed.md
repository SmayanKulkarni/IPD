   
 
17 
 
         4 Proposed System  
 
 
Figure 4.1 : Overall Approach 
 
This section details the journey of a single video frame from raw pixel data to an analyzed 
piece of coaching advice. 
 
Step 1: Player Detection and Isolation 
The primary challenge on a badminton court is ambiguity. To provide feedback to the correct 
person, we must first reliably isolate them from their opponent and any other individuals in 
the frame. 
• Method in Detail: We employ a Region of Interest (ROI) filter, which acts as a spatial 
gate. We configure the system with a vertical threshold (e.g., y < 0.5), instructing it to 
only consider poses whose vertical center is in the top half of the screen. MediaPipe's 
Pose solution is optimized to find a single prominent person; our ROI acts as a crucial 
   
 
18 
 
secondary check to ensure it has locked onto the correct target player and discards any 
detections of the closer, foreground player. This simple but effective heuristic is vital 
for automating the analysis of real-world game footage. 
 
Step 2: 3D Keypoint Extraction 
Simply knowing a player's 2D position on the screen is insufficient for genuine biomechanical 
analysis. To understand limb rotation, depth, and true angles, we require 3D data. 
• Method in Detail: For each valid frame, MediaPipe's BlazePose model is used. It infers 
33  key  body  landmarks.  Crucially,  we  use  the  pose_world_landmarks  output,  which 
provides the joint coordinates in a real-world 3D space, measured in meters from the 
center  of  the  hips.  This  is  a  significant  advantage  over  2D  coordinates  because  it  is 
viewpoint-invariant. An elbow bent at 90 degrees will be calculated as such regardless 
of whether the player is facing the camera, is sideways, or is at an angle. This ensures 
the  analysis  is  robust  and  accurate.  The  output  is  a  sequence  of  vectors,  [frames,  33 
joints, 3 coordinates (x,y,z)], which forms the raw data for our subsequent analysis. 
 
Step 3: Shot Classification 
Providing feedback for a "smash" when the player was attempting a "drop shot" would be 
useless. Therefore, contextual understanding of the action is paramount. 
• Method in Detail: The entire sequence of 3D landmark data is fed into a pre-trained 
Long  Short-Term  Memory  (LSTM)  network.  An  LSTM  was  chosen  over  simpler 
models (like Random Forest) or standard CNNs  because it is specifically designed to 
recognize patterns in sequential data. It maintains an internal "memory" that allows it to 
understand  the  relationship  between  the  pose  in  the  current  frame  and  the  poses  in 
previous  frames.  This  enables  it  to  learn  the  unique  temporal  "signature"  of  each 
badminton shot, for example, the rapid, explosive arm movement of a smash versus the 
slower, more controlled motion of a lift. 
 
Step 4: Posture Analysis and Correction (KSI) 
This is the core of the project, where the numerical data is translated into coaching insights. 
   
 
19 
 
• Method  in  Detail:  Once  the  LSTM  classifies  the  shot  as,  for  example,  a  "forehand 
drive," the system retrieves a pre-recorded, high-quality "expert" sequence for that same 
shot. The user's motion and the expert's motion are then fed into the Kinetic Similarity 
Index (KSI) engine. This engine doesn't just say "correct" or "incorrect"; it provides a 
nuanced score across several dimensions of the  movement. A rule-based system then 
maps these scores to specific, pre-defined feedback statements. For instance, a low score 
in the acceleration component of the KSI will trigger a message focused on generating 
more power. 
 
4.2 Techniques / Methodology 
 
 
4.2.1 The KSI Algorithm: A Deeper Dive 
The  KSI  formula  is  designed  to  mimic  how  an  expert  coach  would  analyze  a  movement:  by 
looking at form, speed, and power as separate but interconnected components. 
Equation (4.1): 
KSI = wp × Spose + wv × Svelocity + wa × Sacceleration 
 
Step 1: Temporal Alignment (DTW) 
• Intuition: Imagine placing two audio tracks of the same song next to each other, but one 
is played slightly faster. If you compare them second-by-second, they will quickly fall 
out of sync. Dynamic Time Warping (DTW) is like a smart audio engineer who stretches 
and compresses the tracks to perfectly align the beats before comparing them. It finds 
the  optimal  frame-to-frame  mapping  between  the  user's  motion  and  the  expert's, 
ensuring we are always comparing corresponding parts of the swing. 
 
Step 2: Calculate Postural Alignment (S_pose) 
• Intuition: This component answers the question: "Is the player making the right 'shapes' 
with their body?" It focuses purely on the angles of the joints, ignoring speed or power. 
• Formula & Rationale: We use Cosine Similarity to compare the vectors of the user's and 
   
 
20 
 
expert's limbs.  
• The formula, Spose 
Equation (4.2): 
 
 
works by isolating the cosine of the angle between two vectors. This brilliantly makes 
the comparison independent of the player's actual limb length, ensuring a tall player 
and a short player can both be judged against the same ideal form. 
 
Step 3: Calculate Velocity Coherence (S_velocity) 
• Intuition: This component answers: "Is the player moving their limbs at the right speed 
and in the right direction?" A player could have perfect form but swing far too slowly. 
• Formula & Rationale: The formula,  
 
Equation (4.3): 
 
 
is a product of two parts. The Cosine Similarity part checks if the user's wrist is moving 
in the same direction as the expert's. The Gaussian Function (e^-x²) part creates a "bell 
curve" score for speed. It gives a perfect score of 1 if the speeds match exactly, and the 
score gracefully decreases the larger the difference in speed, penalizing large deviations 
more heavily than small ones. 
 
Step 4: Calculate Acceleration Profile (S_acceleration) 
• Intuition: This component answers: "Is the player generating force at the right moment?" 
   
 
21 
 
This is the key to explosiveness and power. A badminton smash isn't about moving the 
arm fast the whole time; it's about a rapid acceleration just before impact. 
• Formula & Rationale: The formula 
Equation (4.4): 
       
again  uses  a  Gaussian  Function. It  compares  the  magnitude  of  the  user's  wrist 
acceleration to the expert's at every moment. It will give a high score only if the user 
generates that crucial burst of acceleration at the exact same phase of the swing as the 
expert. 
 
Step 5: Calculate Final KSI Score 
• Intuition: This is the final report card. It combines the individual grades for form, speed, 
and power into a single, overall score. 
• Formula & Rationale: The weighted average,  
KSI=(wp⋅Spose)+(wv⋅Svelocity)+(wa⋅Sacceleration), 
allows for flexibility. For a delicate net shot, you might increase the weight for Spose 
(form is critical). For a powerful smash, you might increase the weights for Svelocity 
and Sacceleration (speed and power are key). This makes the KSI  a highly adaptable 
tool for analyzing different types of shots. 
 
 
4.2.2 Kinematic Thresholding: The Validation Model 
 
This approach defines a "corridor of correctness" from training data and then validates a user's 
motion against it in real-time. 
 
 
Phase 1: Offline Model Building (From Training Data) 
   
 
22 
 
Goal: To define the statistical boundaries of a "correct" movement. 
 
 
 
 
Phase 2: Real-Time User Analysis 
Goal: To check if the user's movement stays within the defined corridor