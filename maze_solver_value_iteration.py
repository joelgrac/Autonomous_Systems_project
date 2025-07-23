#!/usr/bin/env python3
import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge
from collections import deque

ACTIONS = ['N', 'E', 'S', 'W']
DR_DC = {'N': (1, 0), 'E': (0, 1), 'S': (-1, 0), 'W': (0, -1)}
DIRECTIONS = ['N', 'E', 'S', 'W']
DIRECTION_TO_INDEX = {d: i for i, d in enumerate(DIRECTIONS)}

STEP_DISTANCE = 0.000015
LINEAR_SPEED = 0.00004
ANGULAR_SPEED = 0.15
TURN_ANGLE = 1.5708/6
SEARCH_MOVE_DIST = 0.000001

# Example: 10 markers for a 4x5 grid (edit as needed)
MARKER_TO_POS = {
    1: (0, 0),
    2: (0, 2),
    3: (0, 4),
    4: (1, 1),
    5: (1, 3),
    6: (2, 0),
    7: (2, 2),
    8: (2, 4),
    9: (3, 1),
    10: (3, 3),
}
POS_TO_MARKER = {v: k for k, v in MARKER_TO_POS.items()}

class MazeSolverNode:
    def __init__(self):
        rospy.init_node("maze_solver_node")
        self.bridge = CvBridge()
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", TwistStamped, queue_size=1)
        self.image_sub = rospy.Subscriber("/raspicam_node/image_raw", Image, self.image_callback)
        self.dictionary = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        self.parameters = aruco.DetectorParameters_create()
        self.current_orientation = 'N'
        self.search_phase = 0
        self.first_search = True
        self.waiting_for_marker = True
        self.current_marker_id = None
        self.action_queue = deque()
        self.ignore_counter = 0  # <-- Add this for frame ignoring
        self.grid = [
            [0, 0, 0, 0, 0],        # row 0 (south, bottom)
            [0, 0, -100, 0, 0],     # row 1
            [0, 0, 0, -100, 0],     # row 2
            [-100, 0, 100, 0, 0]    # row 3 (north, top)
        ]
        self.policy = self.compute_optimal_policy()
        rospy.loginfo("Ready and waiting for markers.")

    def image_callback(self, msg):
        if self.ignore_counter > 0:  # <-- Ignore frames if counter is set
            self.ignore_counter -= 1
            rospy.loginfo(f"Ignoring frame, {self.ignore_counter} left")
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"CV Bridge error: {e}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)

        if ids is not None:
            ids = ids.flatten()
            for marker_id in ids:
                if marker_id in MARKER_TO_POS:
                    marker_pos = MARKER_TO_POS[marker_id]
                    if self.waiting_for_marker or self.current_marker_id is None:
                        rospy.loginfo(f"Marker {marker_id} detected at {marker_pos}")
                        self.current_marker_id = marker_id
                        self.waiting_for_marker = False
                        # Plan the optimal path from this marker to the next marker (along the optimal policy)
                        next_marker_pos = self.find_next_marker_along_policy(marker_pos)
                        if next_marker_pos == marker_pos:
                            rospy.loginfo("Already at the goal or no further markers.")
                            return
                        actions = self.extract_policy_segment(marker_pos, next_marker_pos)
                        self.action_queue = deque(actions)
                        self.ignore_counter = 10  # <-- Ignore next 10 frames
                        self.execute_next_action(marker_pos)
                    return
        else:
            self.do_search()

    def execute_next_action(self, current_pos):
        if self.action_queue:
            next_action = self.action_queue.popleft()
            self.execute_action(next_action)
            dr, dc = DR_DC[next_action]
            next_pos = (current_pos[0] + dr, current_pos[1] + dc)
            # If next_pos is a marker (but not the goal), stop and wait for marker detection
            if next_pos in POS_TO_MARKER and self.grid[next_pos[0]][next_pos[1]] != 100:
                self.waiting_for_marker = True
                self.current_marker_id = POS_TO_MARKER[next_pos]
                rospy.loginfo(f"Waiting for marker at {next_pos}")
                return
            else:
                rospy.sleep(4)  # <-- Wait 4 seconds between actions
                self.execute_next_action(next_pos)
        else:
            rospy.loginfo("Finished action sequence for this marker.")
            self.waiting_for_marker = True

    def execute_action(self, direction):
        turn = self.calculate_turn(self.current_orientation, direction)
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        if turn == 1:
            msg.twist.angular.z = -ANGULAR_SPEED
            self.cmd_vel_pub.publish(msg)
            rospy.sleep(TURN_ANGLE / ANGULAR_SPEED)
        elif turn == -1:
            msg.twist.angular.z = ANGULAR_SPEED
            self.cmd_vel_pub.publish(msg)
            rospy.sleep(TURN_ANGLE / ANGULAR_SPEED)
        elif abs(turn) == 2:
            msg.twist.angular.z = ANGULAR_SPEED
            self.cmd_vel_pub.publish(msg)
            rospy.sleep(2 * TURN_ANGLE / ANGULAR_SPEED)
        msg.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)
        self.current_orientation = direction
        msg.twist.linear.x = LINEAR_SPEED
        self.cmd_vel_pub.publish(msg)
        rospy.sleep(STEP_DISTANCE / LINEAR_SPEED)
        msg.twist.linear.x = 0.0
        self.cmd_vel_pub.publish(msg)
        rospy.loginfo(f"Moved {direction}")

    def calculate_turn(self, current, target):
        current_idx = DIRECTION_TO_INDEX[current]
        target_idx = DIRECTION_TO_INDEX[target]
        diff = (target_idx - current_idx) % 4
        if diff == 0:
            return 0
        elif diff == 1:
            return 1
        elif diff == 2:
            return 2
        elif diff == 3:
            return -1

    def do_search(self):
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        if self.first_search:
            rospy.loginfo("Waiting 7 seconds before first search movement...")
            rospy.sleep(7)
            self.first_search = False
        if self.search_phase == 0:
            msg.twist.linear.x = -LINEAR_SPEED
            self.cmd_vel_pub.publish(msg)
            rospy.sleep(SEARCH_MOVE_DIST / LINEAR_SPEED)
            msg.twist.linear.x = 0.0
            self.cmd_vel_pub.publish(msg)
            rospy.sleep(10)
            self.search_phase = 1
        elif self.search_phase == 1:
            msg.twist.linear.x = LINEAR_SPEED
            self.cmd_vel_pub.publish(msg)
            rospy.sleep(SEARCH_MOVE_DIST / LINEAR_SPEED)
            msg.twist.linear.x = 0.0
            self.cmd_vel_pub.publish(msg)
            rospy.sleep(10)
            self.search_phase = 0

    def compute_optimal_policy(self):
        grid = self.grid
        rows, cols = len(grid), len(grid[0])
        V = np.zeros((rows, cols))
        policy = {}

        def is_valid(r, c):
            return 0 <= r < rows and 0 <= c < cols and grid[r][c] != -100

        gamma = 0.9
        theta = 0.01

        # Value iteration
        while True:
            delta = 0
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == -100 or grid[r][c] == 100:
                        continue
                    v = V[r][c]
                    values = []
                    for a in ACTIONS:
                        dr, dc = DR_DC[a]
                        nr, nc = r+dr, c+dc
                        if is_valid(nr, nc):
                            values.append(grid[nr][nc] + gamma * V[nr][nc])
                        else:
                            values.append(grid[r][c] + gamma * V[r][c])
                    V[r][c] = max(values)
                    delta = max(delta, abs(v - V[r][c]))
            if delta < theta:
                break

        # Extract optimal action at each cell
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == -100 or grid[r][c] == 100:
                    continue
                best_action = None
                best_value = -np.inf
                for a in ACTIONS:
                    dr, dc = DR_DC[a]
                    nr, nc = r+dr, c+dc
                    if is_valid(nr, nc):
                        value = grid[nr][nc] + gamma * V[nr][nc]
                        if value > best_value:
                            best_value = value
                            best_action = a
                policy[(r, c)] = best_action
        return policy

    def find_next_marker_along_policy(self, start_pos):
        """Follow the optimal policy from start_pos and return the next marker (or goal) position."""
        pos = start_pos
        visited = set()
        while True:
            if self.grid[pos[0]][pos[1]] == 100:
                return pos
            if pos in POS_TO_MARKER and pos != start_pos:
                return pos
            action = self.policy.get(pos)
            if not action:
                return pos
            dr, dc = DR_DC[action]
            next_pos = (pos[0] + dr, pos[1] + dc)
            if next_pos in visited:
                return pos  # prevent loops
            visited.add(next_pos)
            pos = next_pos

    def extract_policy_segment(self, start_pos, end_pos):
        """Return the list of actions along the optimal policy from start_pos to end_pos (not inclusive)."""
        pos = start_pos
        actions = []
        visited = set()
        while pos != end_pos and self.grid[pos[0]][pos[1]] != 100:
            action = self.policy.get(pos)
            if not action:
                break
            actions.append(action)
            dr, dc = DR_DC[action]
            next_pos = (pos[0] + dr, pos[1] + dc)
            if next_pos in visited:
                break  # prevent loops
            visited.add(next_pos)
            pos = next_pos
        return actions

if __name__ == "__main__":
    node = MazeSolverNode()
    rospy.spin()
