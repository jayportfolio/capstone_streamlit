My production solution will be a webapp which can accept features about a London property and display the predicted price 


Step 12: Design Your Deployment Solution
Architecture
Summary
Time Estimate: 2 - 4 hours
You should start deploying a prototype by determining what the deployment would look
like in production. What are the various pieces of the deployment and how do they fit
together? Here are some specific questions to think about:
● What are the major components of your system? What are the inputs and
outputs?
● Where and how will the data be stored?
● How will data get from one component of the system to another?
● What is the lifecycle of your ML/DL model?
○ How frequently do you need to retrain your model? Is it at fixed intervals
when you collect a certain amount of new data or when some other
conditions are met?
○ What kind of data do you need for retraining? How will you store and
manage it?
○ How do you know if the retrained model is good enough to deploy?
This document is authorized for use only by Jaye Holder (jholder10+springboard@protonmail.com). Copying or posting is an
infringement of copyright.
○ How will the retrained model be deployed?
○ How will the retrained model be stored as an artifact?
● How will the system be monitored? How will you debug it if there are problems?
● How will your system respond to unexpected errors or outages?
● What are the specific tools/technologies you’ll use to build this system?
● What is the estimated implementation cost in terms of resources, time, and
money as applicable?