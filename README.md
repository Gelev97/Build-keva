Build-keva

1. Arm 
 - Maestro - control wrist
 - Xarm - control arm
 - Move_Block - python high level coordination control code
 
 2. Computer Vision - Guess block 3D location
  - pictures - test pictures
  - result - test result
  - Params - parameters for block detection
  - detect_block.py - Detect 2d block pictures
  - detect-stack.py - Detect group of block pictures (Build up as a stack)
  
 3. Final
   - This folder is used for the final complete demo
   - Main.py - control the general workflow
   
  4. gazebo
   - Use ROS and GAZEBO to do simulation
   - empty_world - predefined world model
   - gazebo_spawn.py - run it to build certain stacks in GAZEBO
