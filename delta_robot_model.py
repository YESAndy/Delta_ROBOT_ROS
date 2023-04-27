#!/usr/bin/env python3
'''
The Delta Arm Model
'''
import os
import pybullet as p
import numpy as np
import math

class Delta_Robot_Model:

  def __init__(self,base_position = [0,0,0]):
    self.reset(base_position)

  def reset(self, base_position):
    self.buildParamLists()
    urdf_name = os.path.join(os.path.dirname(__file__), "delta_robot_pybullet.urdf")
    self.model_unique_id = p.loadURDF(urdf_name, basePosition=base_position, useFixedBase=True)
    self.buildLookups()
    self.resetLinkFrictions(lateral_friction_coefficient=0)
    self.resetJointsAndMotors()
    self.buildClosedChains()

  def updateStates(self):
    """
    Get states of the robot and the ball, including the position and velocity of:
      - each actuated robot joint
      - the robot end effector
      - the ball
    Return: states: [3 joint_positions, 3 joint velocities, 3 eef positions, 3 eef velocities, 3
    3 ball positions, 3 ball velocities]
    """
    self.joint_pos_dict, self.joint_vel_dict = self.getActuatedJointStates()
    self.eef_pos_arr, self.eef_vel_arr = self.getEndEffectorStates()
    states = np.array(list(self.joint_pos_dict.values()) + list(self.joint_vel_dict.values()) + \
                      self.eef_pos_arr + self.eef_vel_arr)
    return states

  def buildParamLists(self):
      self.leg_num = 3
      self.kp = 1
      self.kd = 0.6
      self.max_motor_force = 150000

      self.end_effector_thickness = 0.01 #thickness of the end effector platform
      self.end_effector_radius = 0.062
      self.upper_leg_names = {1: "upper_leg_1", 2: "upper_leg_2", 3: "upper_leg_3"}
      self.upper_leg_length = 0.368
      self.leg_pos_on_end_effector = {"upper_leg_1": 0.0, "upper_leg_2": 2.0*np.pi/3.0, "upper_leg_3": 4.0*np.pi/3.0} #angular positions of upper legs on the platform
      self.after_spring_joint_vals = {"theta_1": -0, "theta_2": -0, "theta_3": -0} #name of the after_spring_joint: joint_initial_value

      self.end_effector_name = "platform_link"
      self.end_effector_home_vals = {"x": 0.0, "y": 0.0, "z": 0.087}

      # joints to be initialized
      self.init_joint_values = {**self.after_spring_joint_vals, **self.end_effector_home_vals}
      self.motorDict = {**self.after_spring_joint_vals}

  def buildLookups(self):
    """
    Build following look ups in dictionaries:
     1. jointNameToId
     2. linkNameToID
    Note that since each link and its parent joint has the same ID. Base frame by link_id = -1.
    A very important assumption here is in your URDF, you first have a world link, then have a base_link
    """
    # jointNameToId, linkNameToID
    nJoints = p.getNumJoints(self.model_unique_id)
    self.jointNameToId = {}
    self.linkNameToID={}
    for i in range(nJoints):
      jointInfo = p.getJointInfo(self.model_unique_id, i)
      self.jointNameToId[jointInfo[1].decode('UTF-8')] = jointInfo[0]
      self.linkNameToID[jointInfo[12].decode('UTF-8')] = jointInfo[0]

  def resetLinkFrictions(self, lateral_friction_coefficient):
      """
      Reset all links friction
      """
      for id in self.linkNameToID.values():
          p.changeDynamics(bodyUniqueId=self.model_unique_id,
                           linkIndex=id,
                           jointLowerLimit = -1000,
                           jointUpperLimit = 1000,
                           jointLimitForce = 0,
                           lateralFriction=lateral_friction_coefficient,
                           spinningFriction=0.0,
                           rollingFriction=0.0,
                           linearDamping = 0.0,
                           angularDamping = 0.0,
                           jointDamping = 0.0,
                           contactStiffness=0.0,
                           contactDamping=0.0,
                           maxJointVelocity=10000
                           )

  def resetJointsAndMotors(self):
    """
    We do two things here:
      1. Look up the URDF and set the desired joint angles.
      2. Set up motor control: 1. Specify which joints need to be controlled by motors. 2. specify motor control parameters
    """
    self.maxPoint2PointForce = 5000000

    #disable friction in all joints
    for joint_id in self.jointNameToId.values():
        p.setJointMotorControl2(self.model_unique_id, joint_id,
                                controlMode=p.VELOCITY_CONTROL, force=0)

    #All joint values to be initialized
    for joint_name in self.init_joint_values:
      p.resetJointState(bodyUniqueId = self.model_unique_id, jointIndex=self.jointNameToId[joint_name], targetValue=self.init_joint_values[joint_name])


  def buildClosedChains(self):
    """
    Connect links to joints to build closed chain robots, since URDF does not support chain robots.
    """
    joint_axis = [0,0,0]
    for leg_id in range(1, self.leg_num + 1):
        upper_leg_name = self.upper_leg_names[leg_id]
        x = self.end_effector_radius * np.cos(self.leg_pos_on_end_effector[upper_leg_name])
        y = self.end_effector_radius * np.sin(self.leg_pos_on_end_effector[upper_leg_name])
        parent_frame_pos = np.array([ x, y, -self.end_effector_thickness/2.0])  # Cartesian coordnates on the platform, r_platform = 0.062
        child_frame_pos = [self.upper_leg_length, 0.0, 0.0]   # L_upper = 0.368/2.0
        new_joint_id = p.createConstraint(self.model_unique_id, self.linkNameToID[self.end_effector_name],
                                          self.model_unique_id, self.linkNameToID[upper_leg_name],
                                          p.JOINT_POINT2POINT, joint_axis, parent_frame_pos, child_frame_pos)
        p.changeConstraint(new_joint_id, maxForce=self.maxPoint2PointForce)


# ########################## Helpers ##########################
  def setMotorValueByName(self, motorName, desiredValue):
    """
    Joint Position Control using PyBullet's Default joint angle control
    :param motorName: string, motor name
    :param desiredValue: float, angle value
    """
    motorId=self.jointNameToId[motorName]
    p.setJointMotorControl2(bodyIndex=self.model_unique_id,
                          jointIndex=motorId,
                          controlMode=p.POSITION_CONTROL,
                          targetPosition=desiredValue,
                          positionGain=self.kp,
                          velocityGain=self.kd,
                          force=self.max_motor_force)

  def applyJointTorque(self, torqueDict):
      """
      This is reserved for training.
      """
      after_spring_joint_ids = [self.jointNameToId[name] for name in torqueDict]
      p.setJointMotorControlArray(
                      bodyIndex=self.model_unique_id,
                      jointIndices=after_spring_joint_ids,
                      controlMode=p.TORQUE_CONTROL,
                      forces= torqueDict.values()
      )

  def getActuatedJointStates(self):
      joint_pos = {}
      joint_vel = {}
      for joint_name in self.motorDict.keys():
          joint_state = p.getJointState(bodyUniqueId = self.model_unique_id, jointIndex=self.jointNameToId[joint_name])
          joint_pos[joint_name] = joint_state[0]
          joint_vel[joint_name] = joint_state[1]
      return (joint_pos, joint_vel)

  def getEndEffectorStates(self):
      link_state = p.getLinkState(bodyUniqueId = self.model_unique_id, linkIndex = self.linkNameToID[self.end_effector_name], computeLinkVelocity=1)
      link_pos = list(link_state[0])
      link_vel = list(link_state[6])
      return (link_pos, link_vel)


  import math

  def delta_ik(self, x,y,z):
      # Define the robot parameters
      L = 100 # Length of each arm
      r = 50 # Radius of the base
      d = 150 # Distance from the base to the end effector
      
      # Calculate the position of the wrist
      u = x
      v = y
      w = z + d
      
      # Calculate the angles for each arm
      R = math.sqrt(u**2 + v**2)
      phi = math.atan2(v, u)
      alpha = math.atan2(z - d, R)
      beta = math.atan2(math.sqrt(3) * (d - z), R)
      gamma = math.atan2(math.sqrt(3) * (z - d), R)
      
      theta1 = math.atan2(y - (math.cos(phi) * R), x - (math.sin(phi) * R))
      theta2 = math.atan2(math.sqrt(3) * ((math.cos(alpha) * math.cos(phi)) + (math.sin(alpha) * math.sin(phi) * math.cos(beta))) , math.sin(beta))
      theta3 = math.atan2(math.sqrt(3) * ((math.sin(alpha) * math.cos(phi)) - (math.cos(alpha) * math.sin(phi) * math.cos(beta))) , math.sin(beta))
      
      # Convert the joint angles to degrees
      theta1 = math.degrees(theta1)
      theta2 = math.degrees(theta2)
      theta3 = math.degrees(theta3)
      
      # Check if the calculated joint angles are within the range of motion
      if abs(theta1) > 90 or abs(theta2) > 90 or abs(theta3) > 90:
          theta = [float('nan')] # Return NaN if the solution is invalid
      else:
          theta = [theta1, theta2, theta3] # Return the joint angles in degrees
      
      return theta



