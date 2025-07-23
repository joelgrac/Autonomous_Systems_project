Project developed in the context of autonomous navigation, where a robot solves a maze using decision-making techniques based on Markov Decision Processes (MDPs). The Value Iteration approach computes an optimal policy in real-time to navigate the maze, considering rewards, state transitions, and obstacles. The Q-Learning version uses a pre-trained Q-table for the robot to follow a policy learned through trial and error. The system uses ROS for communication with the robot and OpenCV for detection and localization through ArUco markers. The files maze_solver_value_iteration.py and maze_solver_q_learning.py implement these two approaches respectively.

The motor_driver_node.py file implements motor control by converting velocity commands received via ROS into signals for the motors, enabling movement and safe stopping of the robot. A full context, along with the conclusions and final results of the project, can be found in the report (SAUT25_G31.pdf).

Feat Filipa Cunha, João Ferreira and Nádia Gonçalves

